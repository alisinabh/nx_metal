#include <erl_nif.h>
#include <Metal/Metal.h>

typedef struct {
    id<MTLBuffer> buffer;
    char type;
    unsigned int bitsize;
    unsigned int *shape;
    unsigned int elements_count;
} MTLTensorResource;

ErlNifResourceType *buffer_resource_type;
ERL_NIF_TERM to_resource(ErlNifEnv* env, id<MTLBuffer> buffer, char type, unsigned int bitsize, unsigned int *shape, unsigned int elements_count);

const char* metal_type(char type, unsigned int bitsize);
