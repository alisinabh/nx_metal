#include <metal_stdlib>
using namespace metal;

kernel void add_tensors(constant float *a [[buffer(0)]],
                        constant float *b [[buffer(1)]],
                        device float *result [[buffer(2)]],
                        uint tid [[thread_position_in_grid]]) {
  result[tid] = a[tid] + b[tid];
}

constexpr uint max_dimensions = 5; // Maximum number of dimensions supported

// Helper function to broadcast shapes
uint broadcast_shape(constant uint *shape_a, constant uint *shape_b, uint dims,
                     constant uint *shape_size_a, constant uint *shape_size_b,
                     uint *out_shape, uint *stride_a, uint *stride_b) {
    for (uint i = 0; i < dims; i++) {
        if (shape_a[i] == shape_b[i] || shape_a[i] == 1 || shape_b[i] == 1) {
            out_shape[i] = max(shape_a[i], shape_b[i]);
            stride_a[i] = shape_a[i] == 1 ? 0 : shape_size_a[i];
            stride_b[i] = shape_b[i] == 1 ? 0 : shape_size_b[i];
        } else {
            return 1; // Broadcasting is not possible
        }
    }
    return 0; // Broadcasting is possible
}

// Helper function to calculate the index in a tensor based on strides
uint calc_index(constant uint *coords, constant uint *strides, uint dims) {
    uint index = 0;
    for (uint i = 0; i < dims; i++) {
        index += coords[i] * strides[i];
    }
    return index;
}

kernel void add_tensors_with_broadcast(device const float* tensor_a [[buffer(0)]],
                                      device const float* tensor_b [[buffer(1)]],
                                      device float* result [[buffer(2)]],
                                      constant uint &dims [[buffer(3)]],
                                      constant uint *shape_a [[buffer(4)]],
                                      constant uint *shape_b [[buffer(5)]],
                                      constant uint *shape_size_a [[buffer(6)]],
                                      constant uint *shape_size_b [[buffer(7)]],
                                      uint gid [[thread_position_in_grid]]) {
    uint out_shape[max_dimensions];
    uint stride_a[max_dimensions];
    uint stride_b[max_dimensions];
    
    if (broadcast_shape(shape_a, shape_b, dims, shape_size_a, shape_size_b, out_shape, stride_a, stride_b)) {
        // Broadcasting is not possible, so exit the kernel.
        return;
    }
    
    uint total_size = 1;
    for (uint i = 0; i < dims; i++) {
        total_size *= out_shape[i];
    }
    
    if (gid < total_size) {
        uint coords[max_dimensions];
        uint remaining = gid;
        for (uint i = 0; i < dims; i++) {
            coords[i] = remaining / shape_size_a[i];
            remaining = remaining % shape_size_a[i];
        }
        
        uint index_a = calc_index(coords, stride_a, dims);
        uint index_b = calc_index(coords, stride_b, dims);
        
        result[gid] = tensor_a[index_a] + tensor_b[index_b];
    }
}

