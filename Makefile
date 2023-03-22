CFLAGS ?= -O3 -Wall -Wextra -Wno-unused-parameter -fPIC -I/usr/local/include -I$(ERL_EI_INCLUDE_DIR)
LDFLAGS ?= -shared

LDFLAGS += -dynamiclib -undefined dynamic_lookup

SRC_FILES = $(wildcard c_src/*.m)

ifneq ($(MIX_APP_PATH),)
	CFLAGS += -I$(MIX_APP_PATH)/../include
endif

all: priv/tensors_lib.metallib priv/nx_metal_nif.so

priv/tensors_lib.metallib: c_src/tensors_lib.metal
	xcrun -sdk macosx metal -c $< -o priv/tensors_lib.air
	xcrun -sdk macosx metallib priv/tensors_lib.air -o $@
	$(RM) priv/tensors_lib.air

priv/nx_metal_nif.so: $(SRC_FILES)
	$(CC) $(CFLAGS) -x objective-c $(LDFLAGS) -framework Metal -framework Foundation -o $@ $^

clean:
	$(RM) priv/tensors_lib.metallib
	$(RM) priv/nx_metal_nif.so
