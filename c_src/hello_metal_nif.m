#include <erl_nif.h>
#include <Foundation/Foundation.h>
#include <Metal/Metal.h>

#import "MTLBufferReference.h"
#import "MTLDeviceReference.h"


typedef struct {
    MTLDeviceReference *device_ref;
} MTLDeviceResource;

typedef struct {
    MTLBufferReference *buffer_ref;
} MTLBufferResource;

static ErlNifResourceType* device_resource_type;
static ERL_NIF_TERM atom_ok;
static ERL_NIF_TERM atom_error;
static ErlNifResourceType *buffer_resource_type;

static void buffer_resource_dtor(ErlNifEnv *env, void *obj);

static ERL_NIF_TERM hello(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[])
{
    NSString *hello_str = @"Hello, Metal!";
    return enif_make_string(env, [hello_str UTF8String], ERL_NIF_LATIN1);
}

static ERL_NIF_TERM init_metal(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
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
        MTLBufferResource *buffer_res = enif_alloc_resource(buffer_resource_type, sizeof(MTLBufferResource));
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
    MTLBufferResource *buffer_res;
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

static ErlNifFunc nif_funcs[] = {
    {"hello", 0, hello, 0},
    {"init_metal", 0, init_metal, 0},
    {"create_tensor", 2, create_tensor, 0},
    {"tensor_to_list", 1, tensor_to_list, 0}
};


static int on_load(ErlNifEnv* env, void** priv_data, ERL_NIF_TERM load_info) {
    const char* mod = "Elixir.HelloMetalNif";
    device_resource_type = enif_open_resource_type(env, mod, "Device", NULL, ERL_NIF_RT_CREATE, NULL);
    buffer_resource_type = enif_open_resource_type(env, mod, "MTLBufferResource", buffer_resource_dtor, ERL_NIF_RT_CREATE, NULL);

    if (device_resource_type == NULL) {
        return -1;
    }

    atom_ok = enif_make_atom(env, "ok");
    atom_error = enif_make_atom(env, "error");

    return 0;
}

static void buffer_resource_dtor(ErlNifEnv *env, void *obj)
{
    MTLBufferResource *buffer_res = (MTLBufferResource *)obj;
    buffer_res->buffer_ref = nil;
}

ERL_NIF_INIT(Elixir.HelloMetalNif, nif_funcs, on_load, NULL, NULL, NULL)
