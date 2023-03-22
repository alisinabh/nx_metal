#include <metal_stdlib>
using namespace metal;

#define _KERNEL_FUNC_NAME(FUNC_NAME, TYPE) FUNC_NAME ## TYPE

#define ADD_KERNEL_FUNC(TYPE) \
kernel void _KERNEL_FUNC_NAME(add_, TYPE)(constant TYPE *a [[buffer(0)]], \
                      constant TYPE *b [[buffer(1)]], \
                      device TYPE *result [[buffer(2)]], \
                      uint tid [[thread_position_in_grid]]) { \
  result[tid] = a[tid] + b[tid]; \
}

ADD_KERNEL_FUNC(float)
ADD_KERNEL_FUNC(int)
ADD_KERNEL_FUNC(uint)

#undef ADD_KERNEL_FUNC
#undef _KERNEL_FUNC_NAME
