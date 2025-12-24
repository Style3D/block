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

########################################################################################################################
#################################################    intrinsics.py    ##################################################
########################################################################################################################

from .intrinsic import (
	clz,
	block_id,
	block_dim,
	warp_id,
	lane_id,
	grid_dim,
	thread_id,
	threadfence,
	shfl_xor_sync,
)

__all__ = [
	"clz",
	"block_id",
	"block_dim",
	"warp_id",
	"lane_id",
	"grid_dim",
	"thread_id",
	"threadfence",
	"shfl_xor_sync",
]

########################################################################################################################
###################################################    reduce.py    ####################################################
########################################################################################################################

from .reduce import (
	warp_reduce_sum,
	warp_reduce_min,
	warp_reduce_max,
)

__all__ += [
	"warp_reduce_sum",
	"warp_reduce_min",
	"warp_reduce_max",
]

########################################################################################################################
####################################################    aabb.py    #####################################################
########################################################################################################################

from .aabb import (
	Aabb,
	make_aabb,
	make_empty_aabb,
	aabb_merge,
	aabb_center,
	aabb_overlap,
	aabb_surface_area,
	aabb_expand_point,
	aabb_intersect_ray,
	aabb_contains_point,
	aabb_intersect_segment,
)

__all__ += [
	"Aabb",
	"make_aabb",
	"make_empty_aabb",
	"aabb_merge",
	"aabb_center",
	"aabb_overlap",
	"aabb_surface_area",
	"aabb_expand_point",
	"aabb_intersect_ray",
	"aabb_contains_point",
	"aabb_intersect_segment",
]
