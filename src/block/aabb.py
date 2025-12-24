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

########################################################################################################################
######################################################    Aabb    ######################################################
########################################################################################################################

@wp.struct
class Aabb:
    lower: wp.vec3
    upper: wp.vec3

    def __str__(self):
        return f"lower = {self.lower}, upper = {self.upper}"


@wp.func
def make_aabb(lower: wp.vec3, upper: wp.vec3):
    """Create a AABB (initialized with given bounds)"""
    aabb = Aabb()
    aabb.lower = lower
    aabb.upper = upper
    return aabb


@wp.func
def make_empty_aabb():
    """Create an empty AABB (initialized with invalid bounds)"""
    aabb = Aabb()
    aabb.lower = wp.vec3(+wp.inf, +wp.inf, +wp.inf)
    aabb.upper = wp.vec3(-wp.inf, -wp.inf, -wp.inf)
    return aabb


@wp.func
def aabb_center(aabb: Aabb) -> wp.vec3:
    """Compute the center point of the AABB"""
    return 0.5 * (aabb.lower + aabb.upper)


@wp.func
def aabb_surface_area(aabb: Aabb) -> float:
    """Compute the surface area of the AABB (useful for BVH cost evaluation)"""
    d = aabb.upper - aabb.lower
    return 2.0 * (d[0] * d[1] + d[1] * d[2] + d[2] * d[0])


@wp.func
def aabb_merge(aabb_0: Aabb, aabb_1: Aabb):
    """Merge two AABBs into a new one"""
    aabb = Aabb()
    aabb.lower = wp.min(aabb_0.lower, aabb_1.lower)
    aabb.upper = wp.max(aabb_0.upper, aabb_1.upper)
    return aabb


@wp.func
def aabb_expand_point(aabb: Aabb, p: wp.vec3):
    """Expand AABB to include a point"""
    new_aabb = Aabb()
    new_aabb.lower = wp.min(aabb.lower, p)
    new_aabb.upper = wp.max(aabb.upper, p)
    return new_aabb


@wp.func
def aabb_contains_point(aabb: Aabb, p: wp.vec3) -> bool:
    """Check if a point lies inside the AABB (inclusive of boundaries)"""
    return (
        (p[0] >= aabb.lower[0]) and (p[1] >= aabb.lower[1]) and (p[2] >= aabb.lower[2]) and
        (p[0] <= aabb.upper[0]) and (p[1] <= aabb.upper[1]) and (p[2] <= aabb.upper[2])
    )


@wp.func
def aabb_overlap(a: Aabb, b: Aabb) -> bool:
    """Check if two AABBs overlap"""
    return (
		(a.lower[0] <= b.upper[0]) and (a.lower[1] <= b.upper[1]) and (a.lower[2] <= b.upper[2]) and
		(b.lower[0] <= a.upper[0]) and (b.lower[1] <= a.upper[1]) and (b.lower[2] <= a.upper[2])
	)


@wp.func
def aabb_intersect_ray(aabb: Aabb, ray_origin: wp.vec3, inv_dir: wp.vec3):
    """Ray vs. AABB intersection test (slab method)
    Returns (hit: bool, tmin: float, tmax: float).
    """

    t1 = (aabb.lower[0] - ray_origin[0]) * inv_dir[0]
    t2 = (aabb.upper[0] - ray_origin[0]) * inv_dir[0]

    tmin = wp.min(t1, t2)
    tmax = wp.max(t1, t2)

    t1 = (aabb.lower[1] - ray_origin[1]) * inv_dir[1]
    t2 = (aabb.upper[1] - ray_origin[1]) * inv_dir[1]

    tmin = wp.max(tmin, wp.min(t1, t2))
    tmax = wp.min(tmax, wp.max(t1, t2))

    t1 = (aabb.lower[2] - ray_origin[2]) * inv_dir[2]
    t2 = (aabb.upper[2] - ray_origin[2]) * inv_dir[2]

    tmin = wp.max(tmin, wp.min(t1, t2))
    tmax = wp.min(tmax, wp.max(t1, t2))

    hit = (tmax >= wp.max(tmin, 0.0))

    return hit, tmin, tmax


@wp.func
def aabb_intersect_segment(aabb: Aabb, p0: wp.vec3, p1: wp.vec3):
    """Test intersection between a finite line segment and an AABB"""
    ray_dir = p1 - p0
    inv_dir = 1.0 / ray_dir
    hit, tmin, tmax = aabb_intersect_ray(aabb, p0, inv_dir)
    return hit and (tmin <= 1.0 and tmax >= 0.0)
