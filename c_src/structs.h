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
ERL_NIF_TERM wrap(ErlNifEnv* env, MTLTensorResource* resource);

// Floating point precision helpers
__fp16 elem_as___fp16(MTLTensorResource* resource, unsigned long index);
float elem_as_float(MTLTensorResource* resource, unsigned long index);
double elem_as_double(MTLTensorResource* resource, unsigned long index);

// Integer precision helpers
char elem_as_char(MTLTensorResource* resource, unsigned long index);
short elem_as_short(MTLTensorResource* resource, unsigned long index);
int elem_as_int(MTLTensorResource* resource, unsigned long index);
long elem_as_long(MTLTensorResource* resource, unsigned long index);

const char* metal_type(char type, unsigned int bitsize);
