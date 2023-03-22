#include <erl_nif.h>
#include <Metal/Metal.h>

typedef struct {
    id<MTLBuffer> buffer;
    unsigned int bitsize;
    unsigned int *shape;
    unsigned int elements_count;
} MTLTensorResource;

ErlNifResourceType *buffer_resource_type;
ERL_NIF_TERM to_resource(ErlNifEnv* env, id<MTLBuffer> buffer, unsigned int bitsize, unsigned int *shape, unsigned int elements_count);
