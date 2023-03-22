#include <erl_nif.h>
#include <Foundation/Foundation.h>
#include <Metal/Metal.h>

ERL_NIF_TERM atom_ok;
ERL_NIF_TERM atom_error;
void buffer_resource_dtor(ErlNifEnv *env, void *obj);

id<MTLDevice> mtl_device;
id<MTLLibrary> mtl_library;
