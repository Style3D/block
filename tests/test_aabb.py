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
import block as bk

########################################################################################################################
###################################################    test_aabb    ####################################################
########################################################################################################################

def test_aabb():
	# Make empty aabb
	aabb = bk.make_empty_aabb()
	assert aabb.lower == wp.vec3(+wp.inf)
	assert aabb.upper == wp.vec3(-wp.inf)
	
	# Expand point
	aabb = bk.aabb_expand_point(aabb, wp.vec3(1.0, 2.0, 3.0))
	aabb = bk.aabb_expand_point(aabb, wp.vec3(-1.0, 0.0, 4.0))
	assert aabb.lower == wp.vec3(-1.0, 0.0, 3.0)
	assert aabb.upper == wp.vec3(1.0, 2.0, 4.0)
	
	assert bk.aabb_surface_area(aabb) == 16.0
	assert bk.aabb_center(aabb) == wp.vec3(0.0, 1.0, 3.5)
	assert bk.aabb_contains_point(aabb, wp.vec3(0.0, 1.0, 2.0)) == False

	aabb2 = bk.make_aabb(lower=wp.vec3(0.0, 0.0, 0.0), upper=wp.vec3(2.0, 2.0, 2.0))
	assert aabb2.lower == wp.vec3(0.0, 0.0, 0.0)
	assert aabb2.upper == wp.vec3(2.0, 2.0, 2.0)
	
	aabb3 = bk.make_aabb(lower = wp.vec3(-1.0, -1.0, -1.0), upper = wp.vec3(1.0, 1.0, 1.0))
	assert aabb3.lower == wp.vec3(-1.0, -1.0, -1.0)
	assert aabb3.upper == wp.vec3(1.0, 1.0, 1.0)
	
	ray_origin = wp.vec3(0.0, 0.0, -5.0)
	ray_dir = wp.vec3(0.0, 0.0, 1.3)
	inv_dr = 1.0 / ray_dir
	hit, tmin, tmax = bk.aabb_intersect_ray(aabb3, ray_origin, inv_dr)
	assert hit == True
