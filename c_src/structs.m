#import <erl_nif.h>
#import <Metal/Metal.h>

#import "structs.h"

ERL_NIF_TERM to_resource(ErlNifEnv* env, id<MTLBuffer> buffer, unsigned int bitsize, unsigned int *shape, unsigned int elements_count) {
    MTLTensorResource *buffer_res = enif_alloc_resource(buffer_resource_type, sizeof(MTLTensorResource));
    buffer_res->buffer = buffer;
    buffer_res->bitsize = bitsize;
    buffer_res->shape = shape;
    buffer_res->elements_count = elements_count;

    ERL_NIF_TERM buffer_resource_term = enif_make_resource(env, buffer_res);
    enif_release_resource(buffer_res); // Release the resource, the term now holds the reference

    return buffer_resource_term;
}
