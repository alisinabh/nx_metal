#import "nx_metal_nif.h"
#import "structs.h"

static ERL_NIF_TERM metal_device_name(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  return enif_make_string(env, [mtl_device.name UTF8String], ERL_NIF_LATIN1);
}

static ERL_NIF_TERM from_binary(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary binary;
    unsigned int bitsize;
    unsigned int elements_count = 1;
    int shape_dims;
    const ERL_NIF_TERM* shape_tuple_elements;
    char type[2];

    if (!enif_inspect_binary(env, argv[0], &binary) ||
        !enif_get_atom(env, argv[1], type, sizeof(type), ERL_NIF_LATIN1) ||
        !enif_get_uint(env, argv[2], &bitsize) ||
        !enif_get_tuple(env, argv[3], &shape_dims, &shape_tuple_elements)) {
        return enif_make_badarg(env);
    }

    unsigned int *shape = (unsigned int*)enif_alloc(shape_dims * sizeof(unsigned int));
    if (!shape) {
        return enif_raise_exception(env, enif_make_string(env, "Memory allocation failed", ERL_NIF_LATIN1));
    }

    // Read the shape dimensions
    for (int i = 0; i < shape_dims; i++) {
        unsigned int uint_value;
        if (!enif_get_uint(env, shape_tuple_elements[i], &uint_value)) {
            free(shape);
            return enif_make_badarg(env);
        }

        shape[i] = uint_value;
        elements_count *= uint_value;
    }

    unsigned int num_elements = binary.size * 8 / bitsize;

    // Create a Metal buffer
    id<MTLBuffer> buffer = [mtl_device newBufferWithBytes:binary.data
                                                length:num_elements * bitsize / 8
                                               options:MTLResourceStorageModeShared];

    // Create a resource reference and return it
    if (buffer) {
        ERL_NIF_TERM tensor = to_resource(env, buffer, type[0], bitsize, shape, elements_count);
        return enif_make_tuple2(env, atom_ok, tensor);
    } else {
        return atom_error;
    }
}

static ERL_NIF_TERM to_binary(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    MTLTensorResource *buffer_res;
    if (!enif_get_resource(env, argv[0], buffer_resource_type, (void **)&buffer_res)) {
        return enif_make_badarg(env);
    }

    unsigned int limit;
    if (!enif_get_uint(env, argv[1], &limit)) {
        return enif_make_badarg(env);
    }

    id<MTLBuffer> buffer = buffer_res->buffer;
    NSUInteger buffer_length = [buffer length];

    if (limit == 0) {
        limit = buffer_length;
    }

    NSUInteger num_bytes_to_copy = MIN(limit, buffer_length);

    void *buffer_contents = [buffer contents];
    ErlNifBinary binary;
    enif_alloc_binary(num_bytes_to_copy, &binary);
    memcpy(binary.data, buffer_contents, num_bytes_to_copy);

    ERL_NIF_TERM binary_term = enif_make_binary(env, &binary);
    return enif_make_tuple2(env, atom_ok, binary_term);
}

static ERL_NIF_TERM eye(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    unsigned int bitsize;
    unsigned int elements_count = 1;
    int shape_dims;
    const ERL_NIF_TERM* shape_tuple_elements;
    char type[2];

    if (!enif_get_atom(env, argv[0], type, sizeof(type), ERL_NIF_LATIN1) ||
        !enif_get_uint(env, argv[1], &bitsize) ||
        !enif_get_tuple(env, argv[2], &shape_dims, &shape_tuple_elements)) {
        return enif_make_badarg(env);
    }

    unsigned int *shape = (unsigned int*)enif_alloc(shape_dims * sizeof(unsigned int));
    if (!shape) {
        return enif_raise_exception(env, enif_make_string(env, "Memory allocation failed", ERL_NIF_LATIN1));
    }

    unsigned int x_size;
    unsigned int y_size;
    // Read the shape dimensions
    for (int i = 0; i < shape_dims; i++) {
        unsigned int uint_value;
        if (!enif_get_uint(env, shape_tuple_elements[i], &uint_value)) {
            free(shape);
            return enif_make_badarg(env);
        }

        shape[i] = uint_value;
        elements_count *= uint_value;

        // Set size_x and size_y as last two dimension sizes
        x_size = y_size;
        y_size = uint_value;
    }

    const unsigned long size = elements_count * bitsize / 8;

    id<MTLBuffer> buffer = [mtl_device newBufferWithLength:size options:MTLResourceStorageModeShared];
    void *data = [buffer contents];

    unsigned long cursor;
    for(unsigned int i = 0; i < elements_count / (x_size * y_size); i++) {
      for(unsigned int x = 0; x < x_size; x++) {
        for(unsigned int y = 0; y < y_size; y++) {
          cursor = i * x_size * y_size + x * y_size + y;

          if (strcmp("f", type) == 0) {
              switch(bitsize) {
                case 16:
                  ((__fp16 *) data)[cursor] = x == y ? 1.0 : 0.0;
                  break;
                case 32:
                  ((float *) data)[cursor] = x == y ? 1.0 : 0.0;
                  break;
                default:
                  return enif_make_badarg(env);
              }
          } else {
              switch(bitsize) {
                case 8:
                  ((char *) data)[cursor] = x == y ? 1 : 0;
                  break;
                case 16:
                  ((short *) data)[cursor] = x == y ? 1 : 0;
                  break;
                case 32:
                  ((int *) data)[cursor] = x == y ? 1 : 0;
                  break;
                case 64:
                  ((long *) data)[cursor] = x == y ? 1 : 0;
                  break;
                default:
                  return enif_make_badarg(env);
              }
          }
        }
      }
    }

    ERL_NIF_TERM tensor = to_resource(env, buffer, type[0], bitsize, shape, elements_count);
    return enif_make_tuple2(env, atom_ok, tensor);
}

#define AS_TYPE_LOOP(TYPE) \
    for(unsigned int i = 0; i < tensor->elements_count; i++) { \
        ((TYPE *) dstData)[i] = elem_as_ ## TYPE(tensor, i); \
    }

static ERL_NIF_TERM as_type(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    MTLTensorResource *tensor;
    unsigned int bitsize;
    char type[2];

    if (!enif_get_resource(env, argv[0], buffer_resource_type, (void **)&tensor) || 
        !enif_get_atom(env, argv[1], type, sizeof(type), ERL_NIF_LATIN1) ||
        !enif_get_uint(env, argv[2], &bitsize)) {
        return enif_make_badarg(env);
    }

    const unsigned long size = tensor->elements_count * bitsize / 8;
    id<MTLBuffer> destinationBuffer = [mtl_device newBufferWithLength:size options:MTLResourceStorageModeShared];

    void *dstData = [destinationBuffer contents];
    
    if (strcmp("f", type) == 0) {
        switch(bitsize) {
            case 16:
                  AS_TYPE_LOOP(__fp16);
                  break;
            case 32:
                  AS_TYPE_LOOP(float);
                  break;
            default:
                  return enif_make_badarg(env);
        }

    } else {
        switch(bitsize) {
            case 8:
                  AS_TYPE_LOOP(char);
                  break;
            case 16:
                  AS_TYPE_LOOP(short);
                  break;
            case 32:
                  AS_TYPE_LOOP(int);
                  break;
            case 64:
                  AS_TYPE_LOOP(long);
                  break;
            default:
                  return enif_make_badarg(env);
        }
    }

    ERL_NIF_TERM outTensor = to_resource(env, destinationBuffer, type[0], bitsize, tensor->shape, tensor->elements_count);
    return enif_make_tuple2(env, atom_ok, outTensor);
}

#define BIN_OP(OP_NAME) \
static ERL_NIF_TERM nifop_ ## OP_NAME(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) { \
    MTLTensorResource *buffer_a; \
    MTLTensorResource *buffer_b; \
    if (!enif_get_resource(env, argv[0], buffer_resource_type, (void **)&buffer_a) || \
        !enif_get_resource(env, argv[1], buffer_resource_type, (void **)&buffer_b)) { \
        return enif_make_badarg(env); \
    } \
    /* Get metal shader function */ \
    NSString *funName = [NSString stringWithFormat:@#OP_NAME"_%s", metal_type(buffer_a->type, buffer_a->bitsize)]; \
    id<MTLFunction> mtl_function = [mtl_library newFunctionWithName:funName]; \
    if (mtl_function == nil) { \
        NSLog(@"Failed to find the function %@", funName); \
        return atom_error; \
    } \
    /* Create a result buffer */ \
    id<MTLBuffer> result_buffer = [mtl_device newBufferWithLength:buffer_b->buffer.length options:MTLResourceStorageModeShared]; \
    NSError *error = nil; \
    /* Create a compute pipeline state */ \
    id<MTLComputePipelineState> pipeline = [mtl_device newComputePipelineStateWithFunction:mtl_function error:&error]; \
    if (!pipeline) { \
        NSLog(@"Error creating compute pipeline state: %@", error.localizedDescription); \
        return atom_error; \
    } \
    /* Create a command queue, buffer and compute encoder */ \
    id<MTLCommandQueue> commandQueue = [mtl_device newCommandQueue]; \
    id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer]; \
    id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder]; \
    /* Set the pipeline state and buffers */ \
    [computeEncoder setComputePipelineState:pipeline]; \
    [computeEncoder setBuffer:buffer_a->buffer offset:0 atIndex:0]; \
    [computeEncoder setBuffer:buffer_b->buffer offset:0 atIndex:1]; \
    [computeEncoder setBuffer:result_buffer offset:0 atIndex:2]; \
    /* Calculate the number of threads and threadgroups */ \
    MTLSize threadsPerGroup = MTLSizeMake(32, 1, 1); \
    MTLSize numThreadgroups = MTLSizeMake((buffer_a->elements_count + 31) / 32, 1, 1); \
    /* Dispatch the threads */ \
    [computeEncoder dispatchThreadgroups:numThreadgroups threadsPerThreadgroup:threadsPerGroup]; \
    /* End encoding and commit the command buffer */ \
    [computeEncoder endEncoding]; \
    [commandBuffer commit]; \
    [commandBuffer waitUntilCompleted]; \
    /* Wrap the result buffer in a resource and return it */ \
    ERL_NIF_TERM tensor = to_resource(env, result_buffer, buffer_a->type, buffer_a->bitsize, buffer_a->shape, buffer_a->elements_count); \
    return enif_make_tuple2(env, atom_ok, tensor); \
}

BIN_OP(add);
BIN_OP(subtract);
BIN_OP(multiply);
BIN_OP(divide);
BIN_OP(pow);
BIN_OP(remainder);
BIN_OP(atan2);
BIN_OP(min);
BIN_OP(max);
BIN_OP(quotient);

#define BIN_OP_DEF(OP_NAME) {#OP_NAME"", 2, nifop_ ## OP_NAME, 0}

static ErlNifFunc nif_funcs[] = {
    {"metal_device_name", 0, metal_device_name, 0},
    {"from_binary", 4, from_binary, 0},
    {"to_binary", 2, to_binary, 0},
    {"eye", 3, eye, 0},
    {"as_type", 3, as_type, 0},
    BIN_OP_DEF(add),
    BIN_OP_DEF(subtract),
    BIN_OP_DEF(multiply),
    BIN_OP_DEF(divide),
    BIN_OP_DEF(pow),
    BIN_OP_DEF(remainder),
    BIN_OP_DEF(atan2),
    BIN_OP_DEF(min),
    BIN_OP_DEF(max),
    BIN_OP_DEF(quotient),
};

id<MTLLibrary> load_metal_library_from_file(id<MTLDevice> device, const char* file_path) {
    // Read the file into an NSData object
    NSString *path = [NSString stringWithUTF8String:file_path];
    NSError *error = nil;
    NSData *data = [NSData dataWithContentsOfFile:path options:0 error:&error];

    if (!data) {
        NSLog(@"Error reading metallib file: %@", error.localizedDescription);
        return nil;
    }

    // Convert NSData to dispatch_data_t
    dispatch_data_t dispatch_data = dispatch_data_create([data bytes], [data length], NULL, DISPATCH_DATA_DESTRUCTOR_DEFAULT);

    // Create a Metal library from the dispatch_data_t object
    id<MTLLibrary> library = [device newLibraryWithData:dispatch_data error:&error];

    // Release the dispatch_data_t object
    dispatch_release(dispatch_data);

    if (!library) {
        NSLog(@"Error creating Metal library: %@", error.localizedDescription);
    }

    return library;
}

static int on_load(ErlNifEnv* env, void** priv_data, ERL_NIF_TERM load_info) {
    const char* mod = "Elixir.NxMetal.NIF";
    buffer_resource_type = enif_open_resource_type(env, mod, "MTLTensorResource", buffer_resource_dtor, ERL_NIF_RT_CREATE, NULL);

    if (buffer_resource_type == NULL) {
        return -1;
    }

    mtl_device = MTLCreateSystemDefaultDevice();

    if (!mtl_device) {
        NSLog(@"Metal device was not found");
        return -1;
    }

    const char *metallib_path = "priv/tensors_lib.metallib";
    mtl_library = load_metal_library_from_file(mtl_device, metallib_path);
    if (!mtl_library) {
        // Handle the error
        NSLog(@"%s library not found", metallib_path);
        return -1;
    }

    atom_ok = enif_make_atom(env, "ok");
    atom_error = enif_make_atom(env, "error");

    return 0;
}

void buffer_resource_dtor(ErlNifEnv *env, void *obj) {
    MTLTensorResource *buffer_res = (MTLTensorResource *)obj;
    [buffer_res->buffer release];
    enif_release_resource(buffer_res);
}

ERL_NIF_INIT(Elixir.NxMetal.NIF, nif_funcs, on_load, NULL, NULL, NULL)
