#include <erl_nif.h>

static int get_int_from_term(ErlNifEnv *env, ERL_NIF_TERM term, int *data) {
    return enif_get_int(env, term, data);
}

static int *tuple_to_int_array(ErlNifEnv *env, ERL_NIF_TERM tuple, int *array_length) {
    if (!enif_is_tuple(env, tuple)) {
        *array_length = 0;
        return NULL;
    }

    const ERL_NIF_TERM *tuple_elements;
    int tuple_length;

    if (!enif_get_tuple(env, tuple, &tuple_length, &tuple_elements)) {
        *array_length = 0;
        return NULL;
    }

    int *int_array = (int *)malloc(tuple_length * sizeof(int));

    for (int i = 0; i < tuple_length; i++) {
        if (!get_int_from_term(env, tuple_elements[i], &int_array[i])) {
            free(int_array);
            *array_length = 0;
            return NULL;
        }
    }

    *array_length = tuple_length;
    return int_array;
}

