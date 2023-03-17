#include <erl_nif.h>
#include <Foundation/Foundation.h>
#include <Metal/Metal.h>

#import "MTLBufferReference.h"

static ERL_NIF_TERM hello(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[])
{
    NSString *hello_str = @"Hello, Metal!";
    return enif_make_string(env, [hello_str UTF8String], ERL_NIF_LATIN1);
}

static ERL_NIF_TERM init_metal(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[])
{
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (device)
    {
        NSString *deviceName = [device name];
        return enif_make_tuple2(env,
                                enif_make_atom(env, "ok"),
                                enif_make_string(env, [deviceName UTF8String], ERL_NIF_LATIN1));
    }
    else
    {
        return enif_make_atom(env, "error");
    }
}

static ERL_NIF_TERM create_tensor(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[])
{
    // Check if the input is a list
    if (!enif_is_list(env, argv[0]))
    {
        return enif_make_badarg(env);
    }

    // Get the length of the list
    unsigned int length;
    if (!enif_get_list_length(env, argv[0], &length))
    {
        return enif_make_badarg(env);
    }

    // Convert the list to a C array of floats
    float *data = (float *)malloc(length * sizeof(float));
    ERL_NIF_TERM list = argv[0];
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
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    id<MTLBuffer> buffer = [device newBufferWithBytes:data length:length * sizeof(float) options:MTLResourceStorageModeShared];
    free(data);

    // Create a resource reference and return it
    if (buffer)
    {
        MTLBufferReference *buffer_ref = [[MTLBufferReference alloc] initWithBuffer:buffer];
        uintptr_t resource_ptr = (uintptr_t)buffer_ref;
        return enif_make_tuple2(env, enif_make_atom(env, "ok"), enif_make_ulong(env, resource_ptr));
    }
    else
    {
        return enif_make_atom(env, "error");
    }
}

static ERL_NIF_TERM tensor_to_list(ErlNifEnv* env, int argc, const ERL_NIF_TERM argv[]) {
    unsigned long resource_ptr;
    if (!enif_get_ulong(env, argv[0], &resource_ptr)) {
        return enif_make_badarg(env);
    }

    MTLBufferReference *buffer_ref = (MTLBufferReference *)resource_ptr;
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
    {"hello", 0, hello},
    {"init_metal", 0, init_metal},
    {"create_tensor", 1, create_tensor},
    {"tensor_to_list", 1, tensor_to_list}
};

ERL_NIF_INIT(Elixir.HelloMetalNif, nif_funcs, NULL, NULL, NULL, NULL)
