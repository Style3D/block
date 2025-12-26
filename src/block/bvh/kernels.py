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
from ..aabb import Aabb, aabb_merge
from ..reduce import warp_reduce_min, warp_reduce_max
from ..intrinsic import grid_dim, block_dim, lane_id, threadfence, clz

########################################################################################################################
######################################################    Bvh    #######################################################
########################################################################################################################

@wp.kernel
def init_leaf_indices_and_bounds_kernel(
    # outputs
    leaf_indices: wp.array(dtype=int),
    lower_upper: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    
    # Assign each leaf index to its thread ID.
    # This creates a one-to-one mapping between leaf indices and primitives.
    leaf_indices[tid] = tid
    
    # Initialize the global bounding box once (at thread 0).
    # lower_upper[0] will hold the minimum corner (start with +inf),
    # lower_upper[1] will hold the maximum corner (start with -inf).
    if tid == 0:
        lower_upper[0] = wp.vec3(wp.inf)
        lower_upper[1] = wp.vec3(-wp.inf)


@wp.kernel
def eval_scene_aabb_kernel(
    count: int,
    lower_bounds: wp.array(dtype=wp.vec3),
    upper_bounds: wp.array(dtype=wp.vec3),
    # outputs
    lower_upper: wp.array(dtype=wp.vec3),
):
    """
    Compute the axis-aligned bounding box (AABB) for a set of 3D points.
    
    This kernel:
        1. Each thread iterates over a strided subset of vertices and accumulates
           a local min/max bounding box.
        2. A warp-level reduction (`reduce_min` / `reduce_max`) is used to combine
           the local results across threads in the same warp.
        3. The first lane (lane_id == 0) of each warp writes the result into the
           global bounding box using atomic min/max updates.
    
    Args:
        num_verts:   Total number of vertices.
        pos:         Array of 3D vertex positions.
        lower_bound: Output array (single element) storing the global minimum bound.
        upper_bound: Output array (single element) storing the global maximum bound.
    """
    tid = wp.tid()
    
    # Local AABB init
    lower = wp.vec3(wp.inf)      # start with +inf so min works
    upper = wp.vec3(-wp.inf)     # start with -inf so max works
    
    # Strided loop across all vertices
    while tid < count:
        lower = wp.min(lower, lower_bounds[tid])
        upper = wp.max(upper, upper_bounds[tid])
        tid += grid_dim() * block_dim()   # jump by total threads in the grid
    
    # Reduce results across warp
    lower = warp_reduce_min(lower)
    upper = warp_reduce_max(upper)
    
    # One thread per warp writes result to global arrays
    if lane_id() == 0:
        wp.atomic_min(lower_upper, 0, lower)
        wp.atomic_max(lower_upper, 1, upper)


@wp.func
def interleave_double_zero(bits: int) -> int:
    """
    Expands a 10-bit integer into a 30-bit integer by inserting two zero bits
    between each original bit. This is also known as "bit dilation" and is
    a common building block for Morton code (Z-order curve) computation.
    
    Example:
        input:   b9 b8 b7 b6 b5 b4 b3 b2 b1 b0
        output:  b9 0 0 b8 0 0 b7 0 0 ... b0 0 0
    
    Args:
        bits (int): Input integer to be interleaved (typically in [0, 1023]).
    
    Returns:
        int: Interleaved integer with two zeros between each bit.
    """
    bits = (bits | (bits << 16)) & 0xFF0000FF
    bits = (bits | (bits << 8))  & 0x0F00F00F
    bits = (bits | (bits << 4))  & 0xC30C30C3
    return (bits | (bits << 2))  & 0x49249249


@wp.func
def morton_encode(p: wp.vec3) -> int:
    """
    Computes the 3D Morton code (Z-order curve index) for a 3D point in [0,1)^3.
    The Morton code is generated by interleaving the bits of the quantized
    x, y, z coordinates.
    
    - Each coordinate is scaled and quantized to 10 bits (range [0, 1023]).
    - Bit interleaving is performed so that the final Morton code has the
      format: x9 y9 z9 x8 y8 z8 ... x0 y0 z0.
    
    Args:
        p (wp.vec3): 3D point with coordinates in [0,1).
    
    Returns:
        int 30-bit Morton code encoding the 3D position.
    """
    scale_factor = float(1 << (32 // 3))
    p.x = wp.clamp(p.x * scale_factor, 0.0, scale_factor - 1.0)
    p.y = wp.clamp(p.y * scale_factor, 0.0, scale_factor - 1.0)
    p.z = wp.clamp(p.z * scale_factor, 0.0, scale_factor - 1.0)
    x00 = interleave_double_zero(int(p.x))
    y00 = interleave_double_zero(int(p.y))
    z00 = interleave_double_zero(int(p.z))
    return (x00 << 2) + (y00 << 1) + z00


@wp.kernel
def assign_morton_codes_kernel(
    lower_upper: wp.array(dtype=wp.vec3),
    lower_bounds: wp.array(dtype = wp.vec3),
    upper_bounds: wp.array(dtype = wp.vec3),
    # outputs
    morton_codes: wp.array(dtype=int),
):
    """Compute Morton codes (Z-order curve) for all primitives.
    """
    tid = wp.tid()
    
    # Compute the center of the primitive's bounding box
    center = (lower_bounds[tid] + upper_bounds[tid]) * 0.5
    
    # Normalize the center into [0,1] range using the scene AABB
    p = wp.cw_div(center - lower_upper[0], lower_upper[1] - lower_upper[0])
    
    # Encode the normalized position into a Morton code (Z-order curve)
    # Morton codes are used to sort primitives spatially for BVH construction
    morton_codes[tid] = morton_encode(p)


@wp.func
def common_prefix(m0: int, m1: int, i: int, j: int) -> int:
    """Compute common prefix length between two Morton codes.
    If Morton codes are equal, fall back to comparing leaf indices.
    """
    return clz(wp.uint32(m0 ^ m1)) if (m0 != m1) else clz(wp.uint32(i ^ j)) + 32


@wp.func
def delta(m0: int, sorted_morton_codes: wp.array(dtype=int), i: int, j: int, num_leaves: int) -> int:
    """Compute the common prefix length between leaf i and leaf j."""
    return common_prefix(m0, sorted_morton_codes[j], i, j) if (0 <= j < num_leaves) else -1


@wp.func
def determine_range(sorted_morton_codes: wp.array(dtype=int), i: int, num_leaves: int):
    # cache
    m0 = sorted_morton_codes[i]
    
    # Determine direction of the range (+1 or -1)
    d_l = delta(m0, sorted_morton_codes, i, i - 1, num_leaves)
    d_r = delta(m0, sorted_morton_codes, i, i + 1, num_leaves)
    d = wp.sign(d_r - d_l)
    
    # Compute upper bound for the length of the range
    d_min = wp.min(d_l, d_r)
    
    # Expand the range as far as possible
    lmax = wp.int32(2)
    while delta(m0, sorted_morton_codes, i, i + lmax * d, num_leaves) > d_min:
        lmax *= 2
    
    # Find the other end using binary search
    l = wp.int32(0)
    t = lmax // 2
    while t >= 1:
        if delta(m0, sorted_morton_codes, i, i + (l + t) * d, num_leaves) > d_min:
            l = l + t
        t //= 2
    j = i + l * d
    
    return wp.min(i, j), wp.max(i, j)


@wp.func
def find_split(sorted_morton_codes: wp.array(dtype=int), first: int, last: int):
    # Calculate the number of highest bits that are the same
    last_code = sorted_morton_codes[last]
    first_code = sorted_morton_codes[first]
    d_min = common_prefix(first_code, last_code, first, last)
    
    # Use binary search to find where the next bit differs.
    # Specifically, we are looking for the highest object that
    # shares more than commonPrefix bits with the first one.
    split = wp.int32(first)     # initial guess
    step = last - first
    
    while step > 1:
        step = (step + 1) // 2      # exponential decrease
        new_split = split + step    # proposed new position
        if new_split < last:
            if common_prefix(first_code, sorted_morton_codes[new_split], first, new_split) > d_min:
                split = new_split
    
    return split


@wp.kernel
def construct_binary_radix_tree_kernel(
    num_leaves: int,
    sorted_morton_codes: wp.array(dtype=int),
    # outputs
    left_nodes: wp.array(dtype=int),
    right_nodes: wp.array(dtype=int),
    parent_nodes: wp.array(dtype=int),
):
    """
    Reference: Maximizing Parallelism in the Construction of BVHs, Octrees, and k-d Trees. [2012] Tero Karras.
    """
    i = wp.tid()
    first, last = determine_range(sorted_morton_codes, i, num_leaves)
    split = find_split(sorted_morton_codes, first, last)
    
    # Output nodes
    left, right = split, split + 1
    if first == left:       # left child is a leaf
        left += num_leaves - 1
    if last == right:   # right child is a leaf
        right += num_leaves - 1
    
    left_nodes[i] = left
    right_nodes[i] = right
    parent_nodes[left] = i
    parent_nodes[right] = i
    
    if i == 0:
        parent_nodes[0] = -1


@wp.kernel
def verify_binary_radix_tree_kernel(
    num_leaves: int,
    left_nodes: wp.array(dtype = int),
    right_nodes: wp.array(dtype = int),
    parent_nodes: wp.array(dtype = int),
):
    """ Verify a binary radix tree representation.
    """
    tid = wp.tid()
    num_internal_nodes = num_leaves - 1
    num_total_nodes = 2 * num_leaves - 1
    
    if tid < num_internal_nodes:
        left = left_nodes[tid]
        right = right_nodes[tid]
        if not 0 < left < num_total_nodes:
            wp.printf("Error: node %d has invalid left child %d\n", tid, left)
        elif parent_nodes[left] != tid:
            wp.printf("Error: node %d has wrong parent %d\n", tid, parent_nodes[left])
        if not 0 < right < num_total_nodes:
            wp.printf("Error: node %d has invalid right child %d\n", tid, right)
        elif parent_nodes[right] != tid:
            wp.printf("Error: node %d has wrong parent %d\n", tid, parent_nodes[right])
    
    parent = parent_nodes[tid]
    if not -1 <= parent < num_internal_nodes:
        wp.printf("Error: node %d has invalid parent %d\n", tid, parent)
    elif tid == 0 and parent != -1:
        wp.printf("Error: node %d has invalid parent %d\n", tid, parent)
    elif tid != 0 and parent == -1:
        wp.printf("Error: node %d has invalid parent %d\n", tid, parent)
    elif parent != -1:
        if left_nodes[parent] != tid and right_nodes[parent] != tid:
            wp.printf("Error: node %d is not acknowledged by its parent %d :(\n", tid, parent)


@wp.kernel
def assign_escape_indices_kernel(
    num_internal_nodes: int,
    left_nodes: wp.array(dtype=int),
    right_nodes: wp.array(dtype=int),
    # outputs
    escape_indices: wp.array(dtype=int),
):
    """
    """
    tid = wp.tid()
    
    escape_index = wp.int32(-1)
    current_index = wp.int32(0)
    
    if tid < num_internal_nodes:
        current_index = left_nodes[tid]
        escape_index = right_nodes[tid]
    
    escape_indices[current_index] = escape_index
    
    while current_index < num_internal_nodes:
        current_index = right_nodes[current_index]
        escape_indices[current_index] = escape_index


@wp.kernel
def assign_bounding_boxes_kernel(
    num_leaves: int,
    left_nodes: wp.array(dtype=int),
    right_nodes: wp.array(dtype = int),
    parent_nodes: wp.array(dtype = int),
    lower_bounds: wp.array(dtype=wp.vec3),
    upper_bounds: wp.array(dtype=wp.vec3),
    leaf_indcies: wp.array(dtype=int),
    # outputs
    flags: wp.array(dtype = int),
    bound_boxes: wp.array(dtype=Aabb),
):
    tid = wp.tid()
    leaf_index = leaf_indcies[tid]
    current_index = num_leaves - 1 + tid
    parent_index = parent_nodes[current_index]
    
    aabb = Aabb()
    aabb.lower = lower_bounds[leaf_index]
    aabb.upper = upper_bounds[leaf_index]
    bound_boxes[current_index] = aabb
    
    while parent_index != -1:
        threadfence()
        if wp.atomic_add(flags, parent_index, 1) == 1:
            right_child = right_nodes[parent_index]
            left_child = left_nodes[parent_index]
            
            if current_index == left_child:
                aabb = aabb_merge(aabb, bound_boxes[right_child])
            elif current_index == right_child:
                aabb = aabb_merge(aabb, bound_boxes[left_child])
            
            current_index = parent_index
            parent_index = parent_nodes[parent_index]
            bound_boxes[current_index] = aabb
            continue
        return
    # wp.printf("currNode = %d, lower = { %f, %f, %f }, upper = { %f, %f, %f }\n", current_index, aabb.lower.x, aabb.lower.y, aabb.lower.z, aabb.upper.x, aabb.upper.y, aabb.upper.z)


@wp.struct
class BvhNode:
    aabb: Aabb
    left_or_leaf: wp.int32
    escape_index: wp.int32


@wp.kernel
def compact_bvh_nodes_kernel(
    num_leaves: int,
    aabbs: wp.array(dtype=Aabb),
    left_nodes: wp.array(dtype=int),
    leaf_indices: wp.array(dtype=int),
    escape_indicess: wp.array(dtype=int),
    # outputs
    bvh_nodes: wp.array(dtype=BvhNode),
):
    tid = wp.tid()
    bvh_node = BvhNode()
    bvh_node.aabb = aabbs[tid]
    bvh_node.escape_index = escape_indicess[tid]
    
    if tid < num_leaves - 1:
        bvh_node.left_or_leaf = left_nodes[tid]
    else:
        bvh_node.left_or_leaf = -leaf_indices[tid - (num_leaves - 1)] - 1
    
    bvh_nodes[tid] = bvh_node