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
##################################################    test_reduce    ###################################################
########################################################################################################################

def test_reduce():
    
    @wp.kernel
    def reduce_kernel(
        ret_min_int: wp.array(dtype = int),
        ret_min_float: wp.array(dtype = float),
        ret_min_vec3: wp.array(dtype = wp.vec3),
        
        ret_max_int: wp.array(dtype = int),
        ret_max_float: wp.array(dtype = float),
        ret_max_vec3: wp.array(dtype = wp.vec3),
        
        ret_sum_int: wp.array(dtype = int),
        ret_sum_float: wp.array(dtype = float),
        ret_sum_vec3: wp.array(dtype = wp.vec3),
    ):
        tid = wp.tid()
        
        var_int = int(tid)
        ret_min_int[tid] = bk.warp_reduce_min(var_int)
        ret_max_int[tid] = bk.warp_reduce_max(var_int)
        ret_sum_int[tid] = bk.warp_reduce_sum(var_int)
        
        var_float = float(tid)
        ret_min_float[tid] = bk.warp_reduce_min(var_float)
        ret_max_float[tid] = bk.warp_reduce_max(var_float)
        ret_sum_float[tid] = bk.warp_reduce_sum(var_float)
    
        var_vec3 = wp.vec3(wp.float32(tid))
        ret_min_vec3[tid] = bk.warp_reduce_min(var_vec3)
        ret_max_vec3[tid] = bk.warp_reduce_max(var_vec3)
        ret_sum_vec3[tid] = bk.warp_reduce_sum(var_vec3)
    
    dim = 32
    ret_min_int = wp.zeros(dim, dtype = int)
    ret_min_float = wp.zeros(dim, dtype = float)
    ret_min_vec3 = wp.zeros(dim, dtype = wp.vec3)
    
    ret_max_int = wp.zeros(dim, dtype = int)
    ret_max_float = wp.zeros(dim, dtype = float)
    ret_max_vec3 = wp.zeros(dim, dtype = wp.vec3)
    
    ret_sum_int = wp.zeros(dim, dtype = int)
    ret_sum_float = wp.zeros(dim, dtype = float)
    ret_sum_vec3 = wp.zeros(dim, dtype = wp.vec3)
    
    wp.launch(
        reduce_kernel,
        outputs=[
            ret_min_int,
            ret_min_float,
            ret_min_vec3,
            
            ret_max_int,
            ret_max_float,
            ret_max_vec3,
            
            ret_sum_int,
            ret_sum_float,
            ret_sum_vec3,
        ],
        dim = dim,
    )
    
    ret_min_int_np = ret_min_int.numpy()
    ret_min_float_np = ret_min_float.numpy()
    ret_min_vec3_np = ret_min_vec3.numpy()
    
    ret_max_int_np = ret_max_int.numpy()
    ret_max_float_np = ret_max_float.numpy()
    ret_max_vec3_np = ret_max_vec3.numpy()
    
    ret_sum_int_np = ret_sum_int.numpy()
    ret_sum_float_np = ret_sum_float.numpy()
    ret_sum_vec3_np = ret_sum_vec3.numpy()
    
    for i in range(dim):
        assert ret_min_int_np[i] == 0
        assert ret_min_float_np[i] == 0.0
        assert ret_min_vec3_np[i][0] == 0.0
        assert ret_min_vec3_np[i][1] == 0.0
        assert ret_min_vec3_np[i][2] == 0.0
        
        assert ret_max_int_np[i] == 31
        assert ret_max_float_np[i] == 31.0
        assert ret_max_vec3_np[i][0] == 31.0
        assert ret_max_vec3_np[i][1] == 31.0
        assert ret_max_vec3_np[i][2] == 31.0
        
        assert ret_sum_int_np[i] == 496
        assert ret_sum_float_np[i] == 496.0
        assert ret_sum_vec3_np[i][0] == 496.0
        assert ret_sum_vec3_np[i][1] == 496.0
        assert ret_sum_vec3_np[i][2] == 496.0
