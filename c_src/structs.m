#import <stdio.h>
#import <stdlib.h>
#import <string.h>
#import <erl_nif.h>
#import <Metal/Metal.h>

#import "structs.h"

ERL_NIF_TERM to_resource(ErlNifEnv* env, id<MTLBuffer> buffer, char type, unsigned int bitsize, unsigned int *shape, unsigned int elements_count) {
    MTLTensorResource *buffer_res = enif_alloc_resource(buffer_resource_type, sizeof(MTLTensorResource));
    buffer_res->buffer = buffer;
    buffer_res->type = type;
    buffer_res->bitsize = bitsize;
    buffer_res->shape = shape;
    buffer_res->elements_count = elements_count;

    ERL_NIF_TERM buffer_resource_term = enif_make_resource(env, buffer_res);
    enif_release_resource(buffer_res); // Release the resource, the term now holds the reference

    return buffer_resource_term;
}

const char* metal_type(char type, unsigned int bitsize) {
    if (type == 'f') {
        switch (bitsize) {
            case 16:
                return "half";
            case 32:
                return "float";
        }
    } else if (type == 's') {
        switch (bitsize) {
            case 8:
                return "char";
            case 16:
                return "short";
            case 32:
                return "int";
            case 64:
                return "long";
        }
    } else if (type == 'u') {
        switch (bitsize) {
            case 8:
                return "uchar";
            case 16:
                return "ushort";
            case 32:
                return "uint";
            case 64:
                return "ulong";
        }
    }

    return "UNDEFINED";
}
