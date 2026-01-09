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
#############################################    Native CUDA intrinsics    #############################################
########################################################################################################################

@wp.func_native("return gridDim.x;")
def grid_dim() -> int:
    """ Returns the number of blocks in the grid (1D).

    Equivalent to CUDA's `gridDim.x`.

    Note:
        Warp kernels only support 1D launch configuration, so only the x-dimension is meaningful (y = z = 1).
    """
    ...


@wp.func_native("return blockDim.x;")
def block_dim() -> int:
    """ Returns the number of threads per block (1D).

    Equivalent to CUDA's `blockDim.x`.

    Note:
        Warp kernels only support 1D launch configuration, so only the x-dimension is meaningful (y = z = 1).
    """
    ...


@wp.func_native("return blockIdx.x;")
def block_id() -> int:
    """ Returns the block index within the grid (1D).

    Equivalent to CUDA's `blockIdx.x`.

    Note:
        Warp kernels only support 1D launch configuration, so only the x-component is meaningful (y = z = 0).
    """
    ...


@wp.func_native("return threadIdx.x;")
def thread_id() -> int:
    """ Returns the thread index within its block (1D).

    Equivalent to CUDA's `threadIdx.x`.

    Note:
        Warp kernels only support 1D launch configuration, so only the x-component is meaningful (y = z = 0).
    """
    ...


@wp.func_native("""unsigned int laneId; asm("mov.u32 %0, %%warpid;" : "=r"(laneId)); return laneId;""")
def warp_id() -> int:
    """ Returns the SM warp scheduler slot ID of the calling thread (from %warpid).
    """
    ...


@wp.func_native("""unsigned int laneId; asm("mov.u32 %0, %%laneid;" : "=r"(laneId)); return laneId;""")
def lane_id() -> int:
    """ Returns the thread's lane ID within its warp.

    The lane ID is the thread index within a warp (range [0, 31]).
    This value can be used for warp-level programming, e.g., warp shuffles,
    prefix sums, or ballot operations.

    Equivalent to CUDA's:
        unsigned int laneId;
        asm("mov.u32 %0, %%laneid;" : "=r"(laneId));
    """
    ...


@wp.func_native("""return __clz(x);""")
def clz(x: wp.uint32) -> int:
    """ Count Leading Zeros (CLZ).
        Returns the number of leading 0-bits in the 32-bit unsigned integer `x`.
        If `x` is 0, the result is 32.
        Equivalent to CUDA's `__clz()`.
    """
    ...


@wp.func_native("""return __clz((unsigned int)x);""")
def clz(x: wp.int32) -> int:
    """ Count Leading Zeros (CLZ).
        Returns the number of leading 0-bits in the 32-bit signed integer `x`.
        If `x` is 0, the result is 32.
        Equivalent to CUDA's `__clz()`.
    """
    ...


@wp.func_native("""__threadfence();""")
def threadfence():
    """ Device-wide memory fence.
        Ensures that all global and shared memory writes made by the calling thread
        are visible to all threads on the device before any subsequent writes.
        Equivalent to CUDA's `__threadfence()`.
    """
    ...


@wp.func_native("""return __shfl_xor_sync(mask, var, lane_mask, width);""")
def shfl_xor_sync(mask: int, var: int, lane_mask: int, width: int = 32) -> int:
    """
    Warp-level shuffle XOR operation (CUDA intrinsic wrapper) for integers.

    This function allows threads within a warp to exchange values directly
    through registers without using shared memory.
    """
    ...


@wp.func_native("""return __shfl_xor_sync(mask, var, lane_mask, width);""")
def shfl_xor_sync(mask: int, var: float, lane_mask: int, width: int = 32) -> float:
    """
    Warp-level shuffle XOR operation (CUDA intrinsic wrapper) for float-points.

    This function allows threads within a warp to exchange values directly
    through registers without using shared memory.
    """
    ...


@wp.func
def shfl_xor_sync(mask: int, var: wp.vec3, lane_mask: int, width: int = 32) -> wp.vec3:
    """
    Warp-level shuffle XOR operation (CUDA intrinsic wrapper) for 3D vectors.

    This function allows threads within a warp to exchange values directly
    through registers without using shared memory.
    """
    var.x = shfl_xor_sync(mask, var.x, lane_mask, width)
    var.y = shfl_xor_sync(mask, var.y, lane_mask, width)
    var.z = shfl_xor_sync(mask, var.z, lane_mask, width)
    return var


@wp.func_native("""return __int_as_float(val);""")
def int_as_float(val: int) -> float:
    """
    Reinterpret the bit pattern of a 32-bit signed integer as a float.

    This function performs a bitwise reinterpretation without any numeric
    conversion. The input integer bits are returned as a float with the same
    binary representation.
    """
    ...


@wp.func_native("""return __uint_as_float(val);""")
def uint_as_float(val: wp.uint32) -> float:
    """
    Reinterpret the bit pattern of a 32-bit unsigned integer as a float.

    This function performs a bitwise reinterpretation without any numeric
    conversion. The input unsigned integer bits are returned as a float with the same
    binary representation.
    """
    ...


@wp.func_native("""return __float_as_int(val);""")
def float_as_int(val: float) -> int:
    """
    Reinterpret the bit pattern of a float as a 32-bit signed integer.

    This function performs a bitwise reinterpretation without any numeric
    conversion. The input float are returned as an integer with the same
    binary representation.
    """
    ...


@wp.func_native("""return __float_as_uint(val);""")
def float_as_uint(val: float) -> wp.uint32:
    """
    Reinterpret the bit pattern of a float as a 32-bit unsigned integer.
    
    This function performs a bitwise reinterpretation without any numeric
    conversion. The input float are returned as an unsigned integer with the same
    binary representation.
    """
    ...
