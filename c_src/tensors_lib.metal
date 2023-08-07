#include <metal_stdlib>
using namespace metal;

#define KERNEL_FUNC_ALL_TYPES(FUNC_NAME) \
FUNC_NAME(half) \
FUNC_NAME(float) \
FUNC_NAME(char) \
FUNC_NAME(short) \
FUNC_NAME(int) \
FUNC_NAME(long) \
FUNC_NAME(uchar) \
FUNC_NAME(ushort) \
FUNC_NAME(uint) \
FUNC_NAME(ulong)

#define _KERNEL_FUNC_NAME(FUNC_NAME, TYPE) FUNC_NAME ## TYPE

#define ADD_KERNEL_FUNC(TYPE) \
kernel void _KERNEL_FUNC_NAME(add_, TYPE)(constant TYPE *a [[buffer(0)]], \
                      constant TYPE *b [[buffer(1)]], \
                      device TYPE *result [[buffer(2)]], \
                      uint tid [[thread_position_in_grid]]) { \
  result[tid] = a[tid] + b[tid]; \
}

KERNEL_FUNC_ALL_TYPES(ADD_KERNEL_FUNC)

#define SUBTRACT_KERNEL_FUNC(TYPE) \
kernel void _KERNEL_FUNC_NAME(subtract_, TYPE)(constant TYPE *a [[buffer(0)]], \
                      constant TYPE *b [[buffer(1)]], \
                      device TYPE *result [[buffer(2)]], \
                      uint tid [[thread_position_in_grid]]) { \
  result[tid] = a[tid] - b[tid]; \
}

KERNEL_FUNC_ALL_TYPES(SUBTRACT_KERNEL_FUNC)

#define MULTIPLY_KERNEL_FUNC(TYPE) \
kernel void _KERNEL_FUNC_NAME(multiply_, TYPE)(constant TYPE *a [[buffer(0)]], \
                      constant TYPE *b [[buffer(1)]], \
                      device TYPE *result [[buffer(2)]], \
                      uint tid [[thread_position_in_grid]]) { \
  result[tid] = a[tid] * b[tid]; \
}

KERNEL_FUNC_ALL_TYPES(MULTIPLY_KERNEL_FUNC)

#define DIVIDE_KERNEL_FUNC(TYPE) \
kernel void _KERNEL_FUNC_NAME(divide_, TYPE)(constant TYPE *a [[buffer(0)]], \
                      constant TYPE *b [[buffer(1)]], \
                      device TYPE *result [[buffer(2)]], \
                      uint tid [[thread_position_in_grid]]) { \
  result[tid] = a[tid] / b[tid]; \
}

KERNEL_FUNC_ALL_TYPES(DIVIDE_KERNEL_FUNC)

#undef ADD_KERNEL_FUNC
#undef _KERNEL_FUNC_NAME
#undef KERNEL_FUNC_ALL_TYPES
