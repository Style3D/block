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
from block.intrinsic import shfl_xor_sync, lane_id

MASK_FULL = 0xFFFFFFFF

########################################################################################################################
#############################################    Warp-Level Reduce Sum    ##############################################
########################################################################################################################

@wp.func
def warp_reduce_sum(var: int) -> int:
    """
    Warp-level sum reduction for integers.

    Computes the sum of values held by all 32 threads in a warp using
    butterfly-style communication with `shfl_xor_sync`.

    Args:
        var: The integer value from the current thread.

    Returns:
        The sum of all 32 values in the warp. The result is broadcast to
        every thread in the warp, so all threads see the same sum.
    """
    var += shfl_xor_sync(mask = MASK_FULL, var = var, lane_mask = 16)
    var += shfl_xor_sync(mask = MASK_FULL, var = var, lane_mask = 8)
    var += shfl_xor_sync(mask = MASK_FULL, var = var, lane_mask = 4)
    var += shfl_xor_sync(mask = MASK_FULL, var = var, lane_mask = 2)
    var += shfl_xor_sync(mask = MASK_FULL, var = var, lane_mask = 1)
    return var


@wp.func
def warp_reduce_sum(var: float) -> float:
    """
    Warp-level sum reduction for float-points.

    Computes the sum of values held by all 32 threads in a warp using
    butterfly-style communication with `shfl_xor_sync`.

    Args:
        var: The float-point value from the current thread.

    Returns:
        The sum of all 32 values in the warp. The result is broadcast to
        every thread in the warp, so all threads see the same sum.
    """
    var += shfl_xor_sync(mask = MASK_FULL, var = var, lane_mask = 16)
    var += shfl_xor_sync(mask = MASK_FULL, var = var, lane_mask = 8)
    var += shfl_xor_sync(mask = MASK_FULL, var = var, lane_mask = 4)
    var += shfl_xor_sync(mask = MASK_FULL, var = var, lane_mask = 2)
    var += shfl_xor_sync(mask = MASK_FULL, var = var, lane_mask = 1)
    return var


@wp.func
def warp_reduce_sum(var: wp.vec3) -> wp.vec3:
    """
    Warp-level sum reduction for 3D vectors.

    Computes the sum of values held by all 32 threads in a warp using
    butterfly-style communication with `shfl_xor_sync`.

    Args:
        var: The 3D vector value from the current thread.

    Returns:
        The sum of all 32 values in the warp. The result is broadcast to
        every thread in the warp, so all threads see the same sum.
    """
    var += shfl_xor_sync(mask = MASK_FULL, var = var, lane_mask = 16)
    var += shfl_xor_sync(mask = MASK_FULL, var = var, lane_mask = 8)
    var += shfl_xor_sync(mask = MASK_FULL, var = var, lane_mask = 4)
    var += shfl_xor_sync(mask = MASK_FULL, var = var, lane_mask = 2)
    var += shfl_xor_sync(mask = MASK_FULL, var = var, lane_mask = 1)
    return var

########################################################################################################################
#############################################    Warp-Level Reduce Min    ##############################################
########################################################################################################################

@wp.func
def warp_reduce_min(var: int) -> int:
    """
    Warp-level minimum reduction for integer values.

    This function computes the minimum value across all 32 threads in a warp
    using butterfly-style communication with `shfl_xor_sync`.

    Args:
        var: The local integer value from the current thread.

    Returns:
        The minimum of all 32 values in the warp. The result is broadcast to
        every thread in the warp, so all threads will see the same minimum.
    """
    var = wp.min(var, shfl_xor_sync(mask = MASK_FULL, var = var, lane_mask = 16))
    var = wp.min(var, shfl_xor_sync(mask = MASK_FULL, var = var, lane_mask = 8))
    var = wp.min(var, shfl_xor_sync(mask = MASK_FULL, var = var, lane_mask = 4))
    var = wp.min(var, shfl_xor_sync(mask = MASK_FULL, var = var, lane_mask = 2))
    var = wp.min(var, shfl_xor_sync(mask = MASK_FULL, var = var, lane_mask = 1))
    return var


@wp.func
def warp_reduce_min(var: float) -> float:
    """
    Warp-level minimum reduction for float-point values.

    This function computes the minimum value across all 32 threads in a warp
    using butterfly-style communication with `shfl_xor_sync`.

    Args:
        var: The local float-point value from the current thread.

    Returns:
        The minimum of all 32 values in the warp. The result is broadcast to
        every thread in the warp, so all threads will see the same minimum.
    """
    var = wp.min(var, shfl_xor_sync(mask = MASK_FULL, var = var, lane_mask = 16))
    var = wp.min(var, shfl_xor_sync(mask = MASK_FULL, var = var, lane_mask = 8))
    var = wp.min(var, shfl_xor_sync(mask = MASK_FULL, var = var, lane_mask = 4))
    var = wp.min(var, shfl_xor_sync(mask = MASK_FULL, var = var, lane_mask = 2))
    var = wp.min(var, shfl_xor_sync(mask = MASK_FULL, var = var, lane_mask = 1))
    return var


@wp.func
def warp_reduce_min(var: wp.vec3) -> wp.vec3:
    """
    Warp-level minimum reduction for 3D vectors.

    This function computes the minimum value across all 32 threads in a warp
    using butterfly-style communication with `shfl_xor_sync`.

    Args:
        var: The local 3D vector value from the current thread.

    Returns:
        The minimum of all 32 values in the warp. The result is broadcast to
        every thread in the warp, so all threads will see the same minimum.
    """
    var = wp.min(var, shfl_xor_sync(mask = MASK_FULL, var = var, lane_mask = 16))
    var = wp.min(var, shfl_xor_sync(mask = MASK_FULL, var = var, lane_mask = 8))
    var = wp.min(var, shfl_xor_sync(mask = MASK_FULL, var = var, lane_mask = 4))
    var = wp.min(var, shfl_xor_sync(mask = MASK_FULL, var = var, lane_mask = 2))
    var = wp.min(var, shfl_xor_sync(mask = MASK_FULL, var = var, lane_mask = 1))
    return var


########################################################################################################################
#############################################    Warp-Level Reduce Max    ##############################################
########################################################################################################################

@wp.func
def warp_reduce_max(var: int) -> int:
    """
    Warp-level maximum reduction for integer values.

    This function computes the maximum value across all 32 threads in a warp
    using butterfly-style communication with `shfl_xor_sync`.

    Args:
        var: The local integer value from the current thread.

    Returns:
        The maximum of all 32 values in the warp. The result is broadcast to
        every thread in the warp, so all threads will see the same maximum.
    """
    var = wp.max(var, shfl_xor_sync(mask = MASK_FULL, var = var, lane_mask = 16))
    var = wp.max(var, shfl_xor_sync(mask = MASK_FULL, var = var, lane_mask = 8))
    var = wp.max(var, shfl_xor_sync(mask = MASK_FULL, var = var, lane_mask = 4))
    var = wp.max(var, shfl_xor_sync(mask = MASK_FULL, var = var, lane_mask = 2))
    var = wp.max(var, shfl_xor_sync(mask = MASK_FULL, var = var, lane_mask = 1))
    return var


@wp.func
def warp_reduce_max(var: float) -> float:
    """
    Warp-level maximum reduction for float-point values.

    This function computes the maximum value across all 32 threads in a warp
    using butterfly-style communication with `shfl_xor_sync`.

    Args:
        var: The local float-point value from the current thread.

    Returns:
        The maximum of all 32 values in the warp. The result is broadcast to
        every thread in the warp, so all threads will see the same maximum.
    """
    var = wp.max(var, shfl_xor_sync(mask = MASK_FULL, var = var, lane_mask = 16))
    var = wp.max(var, shfl_xor_sync(mask = MASK_FULL, var = var, lane_mask = 8))
    var = wp.max(var, shfl_xor_sync(mask = MASK_FULL, var = var, lane_mask = 4))
    var = wp.max(var, shfl_xor_sync(mask = MASK_FULL, var = var, lane_mask = 2))
    var = wp.max(var, shfl_xor_sync(mask = MASK_FULL, var = var, lane_mask = 1))
    return var


@wp.func
def warp_reduce_max(var: wp.vec3) -> wp.vec3:
    """
    Warp-level maximum reduction for 3D vectors.

    This function computes the maximum value across all 32 threads in a warp
    using butterfly-style communication with `shfl_xor_sync`.

    Args:
        var: The local 3D vector value from the current thread.

    Returns:
        The maximum of all 32 values in the warp. The result is broadcast to
        every thread in the warp, so all threads will see the same maximum.
    """
    var = wp.max(var, shfl_xor_sync(mask = MASK_FULL, var = var, lane_mask = 16))
    var = wp.max(var, shfl_xor_sync(mask = MASK_FULL, var = var, lane_mask = 8))
    var = wp.max(var, shfl_xor_sync(mask = MASK_FULL, var = var, lane_mask = 4))
    var = wp.max(var, shfl_xor_sync(mask = MASK_FULL, var = var, lane_mask = 2))
    var = wp.max(var, shfl_xor_sync(mask = MASK_FULL, var = var, lane_mask = 1))
    return var
