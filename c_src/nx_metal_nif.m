#include <erl_nif.h>
#include <Foundation/Foundation.h>
#include <Metal/Metal.h>

#import "MTLBufferReference.h"
#import "MTLDeviceReference.h"
#import "nif_utils.m"


typedef struct {
    MTLDeviceReference *device_ref;
} MTLDeviceResource;

typedef struct {
    MTLBufferReference *buffer_ref;
    unsigned int bitsize;
    unsigned int *shape;
    unsigned int elements_count;
} MTLTensorResource;

static ErlNifResourceType* device_resource_type;
static ERL_NIF_TERM atom_ok;
static ERL_NIF_TERM atom_error;
static ErlNifResourceType *buffer_resource_type;
static void buffer_resource_dtor(ErlNifEnv *env, void *obj);
static id<MTLDevice> mtl_device;
static id<MTLLibrary> mtl_library;

static ERL_NIF_TERM metal_device_name(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
  return enif_make_string(env, [mtl_device.name UTF8String], ERL_NIF_LATIN1);
}

static ERL_NIF_TERM init_metal_device(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            return enif_make_badarg(env);
        }

        MTLDeviceResource *device_res = enif_alloc_resource(device_resource_type, sizeof(MTLDeviceResource));
        device_res->device_ref = [[MTLDeviceReference alloc] initWithDevice:device];

        ERL_NIF_TERM device_resource_term = enif_make_resource(env, device_res);
        enif_release_resource(device_res); // Release the resource, the term now holds the reference

        NSString *deviceName = device.name;
        ERL_NIF_TERM device_name_term = enif_make_string(env, [deviceName UTF8String], ERL_NIF_LATIN1);

        return enif_make_tuple3(env, atom_ok, device_resource_term, device_name_term);
    }
}

static ERL_NIF_TERM create_tensor(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[])
{
    // Check if the input is a list
    if (!enif_is_list(env, argv[1]))
    {
        return enif_make_badarg(env);
    }

    // Get the device resource from the argument
    MTLDeviceResource *device_res;
    if (!enif_get_resource(env, argv[0], device_resource_type, (void **)&device_res))
    {
        return enif_make_badarg(env);
    }
    id<MTLDevice> device = device_res->device_ref.device;

    // Get the length of the list
    unsigned int length;
    if (!enif_get_list_length(env, argv[1], &length))
    {
        return enif_make_badarg(env);
    }

    // Convert the list to a C array of floats
    float *data = (float *)malloc(length * sizeof(float));
    ERL_NIF_TERM list = argv[1];
    ERL_NIF_TERM head;
    for (unsigned int i = 0; i < length; i++)
    {
        if (!enif_get_list_cell(env, list, &head, &list))
        {
            free(data);
            return enif_make_badarg(env);
        }
        double value;
        if (!enif_get_double(env, head, &value))
        {
            free(data);
            return enif_make_badarg(env);
        }
        data[i] = (float)value;
    }

    // Create a Metal buffer
    id<MTLBuffer> buffer = [device newBufferWithBytes:data length:length * sizeof(float) options:MTLResourceStorageModeShared];
    free(data);

    if (buffer)
    {
        MTLTensorResource *buffer_res = enif_alloc_resource(buffer_resource_type, sizeof(MTLTensorResource));
        buffer_res->buffer_ref = [[MTLBufferReference alloc] initWithBuffer:buffer];

        ERL_NIF_TERM buffer_resource_term = enif_make_resource(env, buffer_res);
        enif_release_resource(buffer_res); // Release the resource, the term now holds the reference

        return enif_make_tuple2(env, atom_ok, buffer_resource_term);
    }
    else
    {
        return atom_error;
    }
}

static ERL_NIF_TERM tensor_to_list(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    MTLTensorResource *buffer_res;
    if (!enif_get_resource(env, argv[0], buffer_resource_type, (void **)&buffer_res)) {
        return enif_make_badarg(env);
    }

    MTLBufferReference *buffer_ref = buffer_res->buffer_ref;
    id<MTLBuffer> buffer = [buffer_ref buffer];
    float *data = (float *)[buffer contents];
    NSUInteger length = [buffer length] / sizeof(float);

    ERL_NIF_TERM list = enif_make_list(env, 0);
    for (NSInteger i = length - 1; i >= 0; i--) {
        list = enif_make_list_cell(env, enif_make_double(env, data[i]), list);
    }

    return list;
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
        MTLTensorResource *buffer_res = enif_alloc_resource(buffer_resource_type, sizeof(MTLTensorResource));
        buffer_res->buffer_ref = [[MTLBufferReference alloc] initWithBuffer:buffer];
        buffer_res->bitsize = bitsize;
        buffer_res->shape = shape;
        buffer_res->elements_count = elements_count;

        ERL_NIF_TERM buffer_resource_term = enif_make_resource(env, buffer_res);
        enif_release_resource(buffer_res); // Release the resource, the term now holds the reference

        return enif_make_tuple2(env, atom_ok, buffer_resource_term);
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

    id<MTLBuffer> buffer = buffer_res->buffer_ref.buffer;
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

    if (buffer)
    {
        MTLTensorResource *buffer_res = enif_alloc_resource(buffer_resource_type, sizeof(MTLTensorResource));
        buffer_res->buffer_ref = [[MTLBufferReference alloc] initWithBuffer:buffer];
        buffer_res->bitsize = bitsize;
        buffer_res->shape = shape;
        buffer_res->elements_count = elements_count;

        ERL_NIF_TERM buffer_resource_term = enif_make_resource(env, buffer_res);
        enif_release_resource(buffer_res); // Release the resource, the term now holds the reference

        return enif_make_tuple2(env, atom_ok, buffer_resource_term);
    }
    else
    {
        return atom_error;
    }
}

static ERL_NIF_TERM add_tensors(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    MTLTensorResource *buffer_a;
    MTLTensorResource *buffer_b;

    if (!enif_get_resource(env, argv[0], buffer_resource_type, (void **)&buffer_a) ||
        !enif_get_resource(env, argv[1], buffer_resource_type, (void **)&buffer_b)) {
        return enif_make_badarg(env);
    }
    
    id<MTLFunction> add_tensors_function = [mtl_library newFunctionWithName:@"add_tensors"];
    if (add_tensors_function == nil) {
        NSLog(@"Failed to find the adder function.");
        return atom_error;
    }

    // Create a result buffer
    id<MTLBuffer> result_buffer = [mtl_device newBufferWithLength:buffer_a->buffer_ref.buffer.length options:MTLResourceStorageModeShared];

     // Create a compute pipeline state
    NSError *error = nil;
    id<MTLComputePipelineState> pipeline = [mtl_device newComputePipelineStateWithFunction:add_tensors_function error:&error];

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
    [computeEncoder setBuffer:buffer_a->buffer_ref.buffer offset:0 atIndex:0];
    [computeEncoder setBuffer:buffer_b->buffer_ref.buffer offset:0 atIndex:1];
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
    MTLTensorResource *buffer_res = enif_alloc_resource(buffer_resource_type, sizeof(MTLTensorResource));
    buffer_res->buffer_ref = [[MTLBufferReference alloc] initWithBuffer:result_buffer];
    buffer_res->bitsize = buffer_a->bitsize;
    buffer_res->shape = buffer_a->shape;
    buffer_res->elements_count = buffer_a->elements_count;

    ERL_NIF_TERM buffer_resource_term = enif_make_resource(env, buffer_res);
    enif_release_resource(buffer_res); // Release the resource, the term now holds the reference

    return enif_make_tuple2(env, atom_ok, buffer_resource_term);
}

static ErlNifFunc nif_funcs[] = {
    {"metal_device_name", 0, metal_device_name, 0},
    {"init_metal_device", 0, init_metal_device, 0},
    {"create_tensor", 2, create_tensor, 0},
    {"tensor_to_list", 1, tensor_to_list, 0},
    {"from_binary", 3, from_binary, 0},
    {"to_binary", 2, to_binary, 0},
    {"eye", 3, eye, 0},
    {"add_tensors", 2, add_tensors, 0}
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
    device_resource_type = enif_open_resource_type(env, mod, "Device", NULL, ERL_NIF_RT_CREATE, NULL);
    buffer_resource_type = enif_open_resource_type(env, mod, "MTLTensorResource", buffer_resource_dtor, ERL_NIF_RT_CREATE, NULL);

    if (device_resource_type == NULL) {
        return -1;
    }

    mtl_device = MTLCreateSystemDefaultDevice();

    if (!mtl_device) {
        NSLog(@"Metal device was not found");
        return -1;
    }

    const char *metallib_path = "c_src/tensors_lib.metallib";
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

static void buffer_resource_dtor(ErlNifEnv *env, void *obj)
{
    MTLTensorResource *buffer_res = (MTLTensorResource *)obj;
    buffer_res->buffer_ref = nil;
}

ERL_NIF_INIT(Elixir.NxMetal.NIF, nif_funcs, on_load, NULL, NULL, NULL)
