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
####################################################    test_clz    ####################################################
########################################################################################################################

def test_clz():
	
	@wp.kernel
	def test_clz_kernel(
		ret_int: wp.array(dtype = int),
		ret_uint: wp.array(dtype = int),
	):
		tid = wp.tid()
		ret_int[tid] = bk.clz(wp.int32(tid))
		ret_uint[tid] = bk.clz(wp.uint32(tid))
	
	dim = 128
	ret_int = wp.zeros(dim, dtype=int)
	ret_uint = wp.zeros(dim, dtype=int)
	
	wp.launch(
		test_clz_kernel,
		outputs = [ret_int, ret_uint],
		dim = dim,
	)
	
	ret_int_np = ret_int.numpy()
	ret_uint_np = ret_uint.numpy()
	
	# Verify result
	for i in range(dim):
		val = i
		ret = 32
		while val != 0:
			ret = ret - 1
			val = val >> 1
		assert ret_int_np[i] == ret_uint_np[i]
		assert ret == ret_int_np[i]

########################################################################################################################
#################################################    test_shfl_xor    ##################################################
########################################################################################################################

def test_shfl_xor():
	
	@wp.kernel
	def test_shfl_xor_kernel(
		ret_int: wp.array(dtype = int),
		ret_float: wp.array(dtype = float),
		ret_vec3: wp.array(dtype = wp.vec3),
	):
		tid = wp.tid()
		
		var_int = int(tid)
		var_int = var_int + bk.shfl_xor_sync(mask = 0xFFFFFFFF, var = var_int, lane_mask = 16)
		var_int = var_int + bk.shfl_xor_sync(mask = 0xFFFFFFFF, var = var_int, lane_mask = 8)
		var_int = var_int + bk.shfl_xor_sync(mask = 0xFFFFFFFF, var = var_int, lane_mask = 4)
		var_int = var_int + bk.shfl_xor_sync(mask = 0xFFFFFFFF, var = var_int, lane_mask = 2)
		var_int = var_int + bk.shfl_xor_sync(mask = 0xFFFFFFFF, var = var_int, lane_mask = 1)
		
		var_float = float(tid)
		var_float = var_float + bk.shfl_xor_sync(mask = 0xFFFFFFFF, var = var_float, lane_mask = 16)
		var_float = var_float + bk.shfl_xor_sync(mask = 0xFFFFFFFF, var = var_float, lane_mask = 8)
		var_float = var_float + bk.shfl_xor_sync(mask = 0xFFFFFFFF, var = var_float, lane_mask = 4)
		var_float = var_float + bk.shfl_xor_sync(mask = 0xFFFFFFFF, var = var_float, lane_mask = 2)
		var_float = var_float + bk.shfl_xor_sync(mask = 0xFFFFFFFF, var = var_float, lane_mask = 1)
		
		var_vec3 = wp.vec3(wp.float32(tid))
		var_vec3 = var_vec3 + bk.shfl_xor_sync(mask = 0xFFFFFFFF, var = var_vec3, lane_mask = 16)
		var_vec3 = var_vec3 + bk.shfl_xor_sync(mask = 0xFFFFFFFF, var = var_vec3, lane_mask = 8)
		var_vec3 = var_vec3 + bk.shfl_xor_sync(mask = 0xFFFFFFFF, var = var_vec3, lane_mask = 4)
		var_vec3 = var_vec3 + bk.shfl_xor_sync(mask = 0xFFFFFFFF, var = var_vec3, lane_mask = 2)
		var_vec3 = var_vec3 + bk.shfl_xor_sync(mask = 0xFFFFFFFF, var = var_vec3, lane_mask = 1)
		
		ret_int[tid] = var_int
		ret_vec3[tid] = var_vec3
		ret_float[tid] = var_float
	
	dim = 32
	ret_int = wp.zeros(dim, dtype = int)
	ret_float = wp.zeros(dim, dtype = float)
	ret_vec3 = wp.zeros(dim, dtype = wp.vec3)
	
	wp.launch(
		test_shfl_xor_kernel,
		outputs = [ret_int, ret_float, ret_vec3],
		dim = dim,
	)
	
	ret_int_np = ret_int.numpy()
	ret_vec3_np = ret_vec3.numpy()
	ret_float_np = ret_float.numpy()
	
	ret = (0 + 31) * dim // 2
	for i in range(dim):
		assert ret_int_np[i] == ret
		assert ret_float_np[i] == float(ret)
		assert ret_vec3_np[i][0] == float(ret)
		assert ret_vec3_np[i][1] == float(ret)
		assert ret_vec3_np[i][2] == float(ret)
		
########################################################################################################################
###############################################    test_launch_config    ###############################################
########################################################################################################################

def test_launch_config():
	
	@wp.kernel
	def test_launch_config_kernel(
		ret_warp_id: wp.array(dtype = int),
		ret_lane_id: wp.array(dtype = int),
		ret_thread_id: wp.array(dtype = int),
		ret_block_id: wp.array(dtype = int),
		ret_block_dim: wp.array(dtype = int),
		ret_grid_dim: wp.array(dtype = int),
	):
		tid = wp.tid()
		ret_warp_id[tid] = bk.warp_id()
		ret_lane_id[tid] = bk.lane_id()
		ret_thread_id[tid] = bk.thread_id()
		ret_block_id[tid] = bk.block_id()
		ret_block_dim[tid] = bk.block_dim()
		ret_grid_dim[tid] = bk.grid_dim()
	
	dim = 128
	ret_warp_id = wp.zeros(dim, dtype = int)
	ret_lane_id = wp.zeros(dim, dtype = int)
	ret_thread_id = wp.zeros(dim, dtype = int)
	ret_block_id = wp.zeros(dim, dtype = int)
	ret_block_dim = wp.zeros(dim, dtype = int)
	ret_grid_dim = wp.zeros(dim, dtype = int)
	
	wp.launch(
		test_launch_config_kernel,
		outputs=[
			ret_warp_id,
			ret_lane_id,
			ret_thread_id,
			ret_block_id,
			ret_block_dim,
			ret_grid_dim,
		],
		block_dim = 64,
		dim=dim,
	)
	
	ret_warp_id_np = ret_warp_id.numpy()
	ret_lane_id_np = ret_lane_id.numpy()
	ret_thread_id_np = ret_thread_id.numpy()
	ret_block_id_np = ret_block_id.numpy()
	ret_block_dim_np = ret_block_dim.numpy()
	ret_grid_dim_np = ret_grid_dim.numpy()
	
	for i in range(dim):
		assert ret_thread_id_np[i] < 64				# random
		assert ret_lane_id_np[i] == ret_thread_id_np[i] % 32
		assert ret_block_id_np[i] < dim // 64		# random
		assert ret_grid_dim_np[i] == dim // 64
		assert ret_block_dim_np[i] == 64

########################################################################################################################
#########################################    test_bitwise_reinterpretation     #########################################
########################################################################################################################

def test_bitwise_reinterpretation():
	
	@wp.kernel
	def test_bitwise_reinterpretation_kernel(
		ret_int: wp.array(dtype = int),
		ret_uint: wp.array(dtype = wp.uint32),
	):
		tid = wp.tid()
		ret_int[tid] = bk.float_as_int(bk.int_as_float(wp.int32(tid)))
		ret_uint[tid] = bk.float_as_uint(bk.uint_as_float(wp.uint32(tid)))
	
	dim = 128
	ret_int = wp.zeros(dim, dtype = int)
	ret_uint = wp.zeros(dim, dtype = wp.uint32)
	
	wp.launch(
		test_bitwise_reinterpretation_kernel,
		inputs = [ret_int, ret_uint],
		dim = dim,
	)
	
	ret_int_np = ret_int.numpy()
	ret_uint_np = ret_uint.numpy()
	
	for i in range(dim):
		assert ret_int_np[i] == i
		assert ret_uint_np[i] == i
	