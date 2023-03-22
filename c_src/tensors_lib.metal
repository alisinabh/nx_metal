#include <metal_stdlib>
using namespace metal;

kernel void add(constant float *a [[buffer(0)]],
                constant float *b [[buffer(1)]],
                device float *result [[buffer(2)]],
                uint tid [[thread_position_in_grid]]) {
  result[tid] = a[tid] + b[tid];
}
