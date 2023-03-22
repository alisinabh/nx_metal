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

    if (!enif_inspect_binary(env, argv[0], &binary) ||
        !enif_get_uint(env, argv[1], &bitsize) ||
        !enif_get_tuple(env, argv[2], &shape_dims, &shape_tuple_elements)) {
        return enif_make_badarg(env);
    }

    unsigned int *shape = (unsigned int*)malloc(shape_dims * sizeof(unsigned int));
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

        ERL_NIF_TERM tensor = to_resource(env, buffer, bitsize, shape, elements_count);
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

    unsigned int *shape = (unsigned int*)malloc(shape_dims * sizeof(unsigned int));
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

    void *data = malloc(size);

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
                case 64:
                  ((double *) data)[cursor] = x == y ? 1.0 : 0.0;
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

    id<MTLBuffer> buffer = [mtl_device newBufferWithBytes:data length:size options:MTLResourceStorageModeShared];
    free(data);

    if (buffer) {
        ERL_NIF_TERM tensor = to_resource(env, buffer, bitsize, shape, elements_count);
        return enif_make_tuple2(env, atom_ok, tensor);
    } else {
        return atom_error;
    }
}

static ERL_NIF_TERM add(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    MTLTensorResource *buffer_a;
    MTLTensorResource *buffer_b;

    if (!enif_get_resource(env, argv[0], buffer_resource_type, (void **)&buffer_a) ||
        !enif_get_resource(env, argv[1], buffer_resource_type, (void **)&buffer_b)) {
        return enif_make_badarg(env);
    }
    
    id<MTLFunction> add_function = [mtl_library newFunctionWithName:@"add_float"];
    if (add_function == nil) {
        NSLog(@"Failed to find the adder function.");
        return atom_error;
    }

    // Create a result buffer
    id<MTLBuffer> result_buffer = [mtl_device newBufferWithLength:buffer_b->buffer.length options:MTLResourceStorageModeShared];

     // Create a compute pipeline state
    NSError *error = nil;
    id<MTLComputePipelineState> pipeline = [mtl_device newComputePipelineStateWithFunction:add_function error:&error];

    if (!pipeline) {
        NSLog(@"Error creating compute pipeline state: %@", error.localizedDescription);
        return atom_error;
    }
// Create a command queue
    id<MTLCommandQueue> commandQueue = [mtl_device newCommandQueue];

    // Create a command buffer
    id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];

    // Create a compute command encoder
    id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];

    // Set the pipeline state and buffers
    [computeEncoder setComputePipelineState:pipeline];
    [computeEncoder setBuffer:buffer_a->buffer offset:0 atIndex:0];
    [computeEncoder setBuffer:buffer_b->buffer offset:0 atIndex:1];
    [computeEncoder setBuffer:result_buffer offset:0 atIndex:2];

    // Calculate the number of threads and threadgroups
    MTLSize threadsPerGroup = MTLSizeMake(16, 1, 1);
    MTLSize numThreadgroups = MTLSizeMake((buffer_a->elements_count + 15) / 16, 1, 1);

    // Dispatch the threads
    [computeEncoder dispatchThreadgroups:numThreadgroups threadsPerThreadgroup:threadsPerGroup];

    // End encoding and commit the command buffer
    [computeEncoder endEncoding];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];

    // Wrap the result buffer in a resource and return it
    ERL_NIF_TERM tensor = to_resource(env, result_buffer, buffer_a->bitsize, buffer_a->shape, buffer_a->elements_count);
    return enif_make_tuple2(env, atom_ok, tensor);
}

static ErlNifFunc nif_funcs[] = {
    {"metal_device_name", 0, metal_device_name, 0},
    {"from_binary", 3, from_binary, 0},
    {"to_binary", 2, to_binary, 0},
    {"eye", 3, eye, 0},
    {"add", 2, add, 0}
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
    buffer_res->buffer = nil;
}

ERL_NIF_INIT(Elixir.NxMetal.NIF, nif_funcs, on_load, NULL, NULL, NULL)
