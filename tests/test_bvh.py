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
import numpy as np

########################################################################################################################
####################################################    test_bvh    ####################################################
########################################################################################################################

def test_bvh():
   radius = 1e-2
   count = 1000000
   np.random.seed(0)
   rand_nums = np.random.rand(3 * count)
   lower_bounds = wp.array(rand_nums - radius, dtype = wp.vec3)
   upper_bounds = wp.array(rand_nums + radius, dtype = wp.vec3)
   
   bvh = bk.Bvh(lower_bounds, upper_bounds)
   bvh.rebuild()
   bvh.refit()
   
   aabb = bvh.bound_boxes.numpy()[0]
   lower, upper = aabb[0], aabb[1]
   assert lower[0] >= 0.0 - radius * 2
   assert lower[1] >= 0.0 - radius * 2
   assert lower[2] >= 0.0 - radius * 2
   assert upper[0] <= 1.0 + radius * 2
   assert upper[1] <= 1.0 + radius * 2
   assert upper[2] <= 1.0 + radius * 2
