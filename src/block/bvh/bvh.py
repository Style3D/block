# BSD 3-Clause License
#
# Copyright (c) 2025, Style3D
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import warp as wp
from ..aabb import Aabb
from .kernels import (
    BvhNode,
    eval_scene_aabb_kernel,
    compact_bvh_nodes_kernel,
    assign_morton_codes_kernel,
    assign_escape_indices_kernel,
    assign_bounding_boxes_kernel,
    verify_binary_radix_tree_kernel,
    construct_binary_radix_tree_kernel,
    init_leaf_indices_and_bounds_kernel,
)

########################################################################################################################
######################################################    Bvh    #######################################################
########################################################################################################################

class Bvh:
    """
    Bounding Volume Hierarchy (BVH) built on GPU using Warp.
    
    This class constructs and maintains a linearized BVH for a set of axis-aligned
    bounding boxes (AABBs), typically used for spatial acceleration in ray tracing,
    collision detection, or nearest-neighbor queries.
    
    The BVH is built using:
        - Morton code sorting for leaf ordering
        - Binary radix tree construction
        - Bottom-up bounding box refitting
        - Linearized node layout with escape indices
        
    All data is stored on the same Warp device (CUDA only).
    """
    
    def __init__(
        self,
        lower_bounds: wp.array(dtype = wp.vec3),
        upper_bounds: wp.array(dtype = wp.vec3),
    ):
        """
        Construct a BVH from per-leaf axis-aligned bounding boxes.
        
        Parameters
        ----------
        lower_bounds : wp.array(dtype=wp.vec3)
            Array of minimum corners (AABB min) for each leaf.
            Shape: (N,)
        
        upper_bounds : wp.array(dtype=wp.vec3)
            Array of maximum corners (AABB max) for each leaf.
            Shape: (N,)
        
        Notes
        -----
        - `lower_bounds` and `upper_bounds` must:
            * Have the same length
            * Reside on the same Warp device
        - Each element pair (lower_bounds[i], upper_bounds[i]) defines one leaf AABB.
        - The BVH is immediately built during construction via `self.rebuild()`.
        """
        if len(lower_bounds) != len(upper_bounds):
            raise ValueError(
                "lower_bounds and upper_bounds must have the same length "
                f"(got {len(lower_bounds)} and {len(upper_bounds)})"
            )
        if lower_bounds.device != upper_bounds.device:
            raise ValueError(
                "lower_bounds and upper_bounds must be on the same device "
                f"(got {lower_bounds.device} and {upper_bounds.device})"
            )
        if not lower_bounds.device.is_cuda:
            raise ValueError(
                "Bvh requires a CUDA device, "
                f"but got device '{lower_bounds.device}'"
            )
        
        # Input leaf AABBs
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        
        # Warp device (CUDA)
        self.device = lower_bounds.device
        
        # Leaf and node counts
        self.num_leaves = len(lower_bounds)
        self.num_internal_nodes = wp.max(0, self.num_leaves - 1)
        self.num_total_nodes = wp.max(0, 2 * self.num_leaves - 1)
        
        # Internal synchronization flags used during bottom-up refit
        self._flags = wp.zeros(self.num_internal_nodes, dtype = int, device = self.device)
        
        # Scene-level bounding box (min/max packed into a length-2 array)
        self._scene_lower_upper = wp.zeros(2, dtype = wp.vec3, device = self.device)
        
        # Leaf indices after Morton sorting (double-buffered)
        self._leaf_indices = wp.zeros(self.num_leaves * 2, dtype = int, device = self.device)
        
        # Morton codes for leaves (double-buffered)
        self._morton_codes = wp.zeros(self.num_leaves * 2, dtype = int, device = self.device)
        
        # Bounding boxes for all nodes (internal + leaf)
        self.bound_boxes = wp.zeros(self.num_total_nodes, dtype = Aabb, device = self.device)
        
        # Parent index for each node (-1 for root)
        self.parent_nodes = wp.zeros(self.num_total_nodes, dtype = int, device = self.device)
        
        # Child indices for internal nodes
        self.left_nodes = wp.zeros(self.num_internal_nodes, dtype = int, device = self.device)
        self.right_nodes = wp.zeros(self.num_internal_nodes, dtype = int, device = self.device)
        
        # Escape indices for stackless traversal
        self.escape_indices = wp.zeros(self.num_total_nodes, dtype = int, device = self.device)
        
        # Final compact linear BVH node array
        self.bvh_nodes = wp.zeros(self.num_total_nodes, dtype = BvhNode, device = self.device)
        
        # Immediately build
        self.rebuild()
    
    def rebuild(self):
        """
        Rebuild the entire BVH from scratch.
        
        This function performs the full BVH construction pipeline:
            1. Initialize leaf indices and per-leaf bounds
            2. Compute the scene-level bounding box
            3. Assign Morton codes to leaves
            4. Sort leaves by Morton code
            5. Construct a binary radix tree
            6. Compute escape indices for stackless traversal
            7. Refit bounding boxes bottom-up
            8. Compact nodes into a linear BVH layout
        
        Notes
        -----
        - This function is GPU-heavy and should be called sparingly.
        - For dynamic scenes where leaf bounds change but topology remains,
          prefer calling `refit()` instead.
        - All kernels are launched on `self.device`.
        """
        # ---------------------------------------------------------------------
        # Early exit: empty BVH (no leaves)
        # ---------------------------------------------------------------------
        if self.num_leaves == 0:
            return
        
        # ---------------------------------------------------------------------
        # 1. Initialize leaf indices and reset scene-level AABB (min/max)
        #    - leaf_indices[i] = i
        #    - scene_lower_upper = {+inf, -inf}
        # ---------------------------------------------------------------------
        wp.launch(
            init_leaf_indices_and_bounds_kernel,
            outputs = [self._leaf_indices, self._scene_lower_upper],
            dim = self.num_leaves,
            device = self.device,
        )
        
        # ---------------------------------------------------------------------
        # 2. Compute scene-level AABB by reducing all leaf bounds
        #    - Parallel reduction over (lower_bounds, upper_bounds)
        #    - Launch size is adapted to SM count for better occupancy
        # ---------------------------------------------------------------------
        wp.launch(
            eval_scene_aabb_kernel,
            dim = wp.min(self.device.sm_count * 2, (self.num_leaves + 63) // 64) * 64,
            inputs = [self.num_leaves, self.lower_bounds, self.upper_bounds],
            outputs = [self._scene_lower_upper],
            device = self.device,
            block_dim = 64,
        )
        # print(self._lower_upper.numpy())
        
        # ---------------------------------------------------------------------
        # 3. Assign Morton codes to each leaf based on scene-normalized AABB
        #    - Maps 3D positions to 1D Morton order
        #    - Used for spatially coherent leaf ordering
        # ---------------------------------------------------------------------
        wp.launch(
            assign_morton_codes_kernel,
            inputs = [self._scene_lower_upper, self.lower_bounds, self.upper_bounds],
            outputs = [self._morton_codes],
            dim = self.num_leaves,
            device = self.device,
        )
        
        # ---------------------------------------------------------------------
        # 4. Sort leaves by Morton code
        #    - Produces spatially ordered leaf indices
        #    - Uses Warp's device-side radix sort
        # ---------------------------------------------------------------------
        if self.num_leaves > 0:
            wp.context.runtime.core.wp_radix_sort_pairs_int_device(
                self._morton_codes.ptr,
                self._leaf_indices.ptr,
                self.num_leaves,
            )
            # print(self._morton_codes.numpy()[:self.num_leaves])
        
        # ---------------------------------------------------------------------
        # 5. Construct binary radix tree from sorted Morton codes
        #    - Generates parent / left / right child relationships
        #    - One internal node per adjacent Morton code interval
        # ---------------------------------------------------------------------
        if self.num_leaves > 1:
            wp.launch(
                construct_binary_radix_tree_kernel,
                inputs = [self.num_leaves, self._morton_codes],
                outputs = [self.left_nodes, self.right_nodes, self.parent_nodes],
                dim = self.num_internal_nodes,
                device = self.device,
            )
        else:
            # Single-leaf BVH: no internal nodes
            self.parent_nodes.fill_(-1)
        
        # print(f"left_nodes = {self.left_nodes}")
        # print(f"right_nodes = {self.right_nodes}")
        # print(f"parent_nodes = {self.parent_nodes}")
        
        # ---------------------------------------------------------------------
        # 6. (Debug) Verify binary radix tree topology
        #    - Checks parent/child consistency
        #    - Intended for development and debugging
        # ---------------------------------------------------------------------
        if True:
            wp.launch(
                verify_binary_radix_tree_kernel,
                inputs = [self.num_leaves, self.left_nodes, self.right_nodes, self.parent_nodes],
                dim = self.num_total_nodes,
                device = self.device,
            )
        
        # ---------------------------------------------------------------------
        # 7. Compute escape indices for stackless BVH traversal
        #    - Enables iterative traversal without an explicit stack
        # ---------------------------------------------------------------------
        wp.launch(
            assign_escape_indices_kernel,
            inputs = [self.num_internal_nodes, self.left_nodes, self.right_nodes],
            outputs = [self.escape_indices],
            dim = self.num_leaves,
            device = self.device,
        )
        # print(self.escape_indices)
        
        # ---------------------------------------------------------------------
        # 8. Refit bounding boxes and compact nodes into linear BVH layout
        # ---------------------------------------------------------------------
        self.refit()
    
    def refit(self):
        """
        Refit the BVH bounding boxes without rebuilding topology.
        
        This function updates internal node bounding boxes assuming:
            - The BVH topology (tree structure) is unchanged
            - Leaf AABBs (`lower_bounds`, `upper_bounds`) may have changed
        
        It performs a bottom-up refit using atomic/synchronization flags.
        
        Notes
        -----
        - This is significantly cheaper than `rebuild()`.
        - Must not be called if the number of leaves has changed.
        - Safe for dynamic scenes with moving geometry.
        """
        # ---------------------------------------------------------------------
        # Early exit: empty BVH (no leaves)
        # ---------------------------------------------------------------------
        if self.num_leaves == 0:
            return
        
        # ---------------------------------------------------------------------
        # Reset synchronization flags for bottom-up bounding box refit
        # ---------------------------------------------------------------------
        self._flags.fill_(0)
        
        # ---------------------------------------------------------------------
        # 1. Bottom-up bounding box assignment
        #    - Initializes leaf node AABBs from input bounds
        #    - Propagates bounding boxes up the tree
        #    - Uses atomic flags to synchronize sibling completion
        # ---------------------------------------------------------------------
        wp.launch(
            assign_bounding_boxes_kernel,
            inputs = [
                self.num_leaves,
                self.left_nodes,
                self.right_nodes,
                self.parent_nodes,
                self.lower_bounds,
                self.upper_bounds,
                self._leaf_indices,
            ],
            outputs = [self._flags, self.bound_boxes],
            dim = self.num_leaves,
            device = self.device,
        )
        
        # ---------------------------------------------------------------------
        # 2. Compact BVH nodes into a linear, traversal-friendly layout
        #    - Reorders nodes into a contiguous array
        #    - Produces final BvhNode representation
        #    - Uses escape indices for stackless traversal
        # ---------------------------------------------------------------------
        wp.launch(
            compact_bvh_nodes_kernel,
            inputs = [
                self.num_leaves,
                self.bound_boxes,
                self.left_nodes,
                self._leaf_indices,
                self.escape_indices,
            ],
            outputs = [self.bvh_nodes],
            dim = self.num_total_nodes,
            device = self.device,
        )
