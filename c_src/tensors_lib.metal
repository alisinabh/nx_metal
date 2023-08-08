#include <metal_stdlib>
using namespace metal;

#define KERNEL_FUNC_ALL_TYPES(FUNC_NAME) \
KERNEL_FUNC_FLOATING_TYPES(FUNC_NAME) \
KERNEL_FUNC_INT_TYPES(FUNC_NAME)

#define KERNEL_FUNC_FLOATING_TYPES(FUNC_NAME) \
FUNC_NAME(half) \
FUNC_NAME(float)

#define KERNEL_FUNC_INT_TYPES(FUNC_NAME) \
FUNC_NAME(char) \
FUNC_NAME(short) \
FUNC_NAME(int) \
FUNC_NAME(long) \
FUNC_NAME(uchar) \
FUNC_NAME(ushort) \
FUNC_NAME(uint) \
FUNC_NAME(ulong)

#define _KERNEL_FUNC_NAME(FUNC_NAME, TYPE) FUNC_NAME ## TYPE

// Binary operations

// add
#define ADD_KERNEL_FUNC(TYPE) \
kernel void _KERNEL_FUNC_NAME(add_, TYPE)(constant TYPE *a [[buffer(0)]], \
                      constant TYPE *b [[buffer(1)]], \
                      device TYPE *result [[buffer(2)]], \
                      uint tid [[thread_position_in_grid]]) { \
  result[tid] = a[tid] + b[tid]; \
}

KERNEL_FUNC_ALL_TYPES(ADD_KERNEL_FUNC)

// subtract
#define SUBTRACT_KERNEL_FUNC(TYPE) \
kernel void _KERNEL_FUNC_NAME(subtract_, TYPE)(constant TYPE *a [[buffer(0)]], \
                      constant TYPE *b [[buffer(1)]], \
                      device TYPE *result [[buffer(2)]], \
                      uint tid [[thread_position_in_grid]]) { \
  result[tid] = a[tid] - b[tid]; \
}

KERNEL_FUNC_ALL_TYPES(SUBTRACT_KERNEL_FUNC)

// multiply
#define MULTIPLY_KERNEL_FUNC(TYPE) \
kernel void _KERNEL_FUNC_NAME(multiply_, TYPE)(constant TYPE *a [[buffer(0)]], \
                      constant TYPE *b [[buffer(1)]], \
                      device TYPE *result [[buffer(2)]], \
                      uint tid [[thread_position_in_grid]]) { \
  result[tid] = a[tid] * b[tid]; \
}

KERNEL_FUNC_ALL_TYPES(MULTIPLY_KERNEL_FUNC)

// divide
#define DIVIDE_KERNEL_FUNC(TYPE) \
kernel void _KERNEL_FUNC_NAME(divide_, TYPE)(constant TYPE *a [[buffer(0)]], \
                      constant TYPE *b [[buffer(1)]], \
                      device TYPE *result [[buffer(2)]], \
                      uint tid [[thread_position_in_grid]]) { \
  result[tid] = a[tid] / b[tid]; \
}

KERNEL_FUNC_ALL_TYPES(DIVIDE_KERNEL_FUNC)

// pow
#define POW_KERNEL_FUNC(TYPE) \
kernel void _KERNEL_FUNC_NAME(pow_, TYPE)(constant TYPE *a [[buffer(0)]], \
                      constant TYPE *b [[buffer(1)]], \
                      device TYPE *result [[buffer(2)]], \
                      uint tid [[thread_position_in_grid]]) { \
  result[tid] = pow((float)a[tid], (float)b[tid]); \
}

KERNEL_FUNC_ALL_TYPES(POW_KERNEL_FUNC)

// remainder
#define REMAINDER_KERNEL_FUNC(TYPE) \
kernel void _KERNEL_FUNC_NAME(remainder_, TYPE)(constant TYPE *a [[buffer(0)]], \
                      constant TYPE *b [[buffer(1)]], \
                      device TYPE *result [[buffer(2)]], \
                      uint tid [[thread_position_in_grid]]) { \
  result[tid] = a[tid] % b[tid]; \
}

KERNEL_FUNC_INT_TYPES(REMAINDER_KERNEL_FUNC)

kernel void remainder_half(constant half *a [[buffer(0)]],
                      constant half *b [[buffer(1)]],
                      device half *result [[buffer(2)]],
                      uint tid [[thread_position_in_grid]]) {
  result[tid] = (int)a[tid] % (int)b[tid];
}

kernel void remainder_float(constant float *a [[buffer(0)]],
                      constant float *b [[buffer(1)]],
                      device float *result [[buffer(2)]],
                      uint tid [[thread_position_in_grid]]) {
  result[tid] = (int)a[tid] % (int)b[tid];
}

// atan2
#define ATAN2_KERNEL_FUNC(TYPE) \
kernel void _KERNEL_FUNC_NAME(atan2_, TYPE)(constant TYPE *a [[buffer(0)]], \
                      constant TYPE *b [[buffer(1)]], \
                      device TYPE *result [[buffer(2)]], \
                      uint tid [[thread_position_in_grid]]) { \
  result[tid] = atan2(float(a[tid]), float(b[tid])); \
}

KERNEL_FUNC_ALL_TYPES(ATAN2_KERNEL_FUNC)

// min
#define MIN_KERNEL_FUNC(TYPE) \
kernel void _KERNEL_FUNC_NAME(min_, TYPE)(constant TYPE *a [[buffer(0)]], \
                      constant TYPE *b [[buffer(1)]], \
                      device TYPE *result [[buffer(2)]], \
                      uint tid [[thread_position_in_grid]]) { \
  result[tid] = min(a[tid], b[tid]); \
}

KERNEL_FUNC_ALL_TYPES(MIN_KERNEL_FUNC)

// max
#define MAX_KERNEL_FUNC(TYPE) \
kernel void _KERNEL_FUNC_NAME(max_, TYPE)(constant TYPE *a [[buffer(0)]], \
                      constant TYPE *b [[buffer(1)]], \
                      device TYPE *result [[buffer(2)]], \
                      uint tid [[thread_position_in_grid]]) { \
  result[tid] = max(a[tid], b[tid]); \
}

KERNEL_FUNC_ALL_TYPES(MAX_KERNEL_FUNC)

// quotient
#define QUOTIENT_KERNEL_FUNC(TYPE) \
kernel void _KERNEL_FUNC_NAME(quotient_, TYPE)(constant TYPE *a [[buffer(0)]], \
                      constant TYPE *b [[buffer(1)]], \
                      device TYPE *result [[buffer(2)]], \
                      uint tid [[thread_position_in_grid]]) { \
  result[tid] = a[tid] / b[tid]; \
}

KERNEL_FUNC_INT_TYPES(QUOTIENT_KERNEL_FUNC)

#undef ADD_KERNEL_FUNC
#undef _KERNEL_FUNC_NAME
#undef KERNEL_FUNC_ALL_TYPES
