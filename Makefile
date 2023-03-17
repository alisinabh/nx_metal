CFLAGS ?= -O3 -Wall -Wextra -Wno-unused-parameter -fPIC -I/usr/local/include -I$(ERL_EI_INCLUDE_DIR)
LDFLAGS ?= -shared

LDFLAGS += -dynamiclib -undefined dynamic_lookup

ifneq ($(MIX_APP_PATH),)
	CFLAGS += -I$(MIX_APP_PATH)/../include
endif

all: priv/nx_metal_nif.so

priv/nx_metal_nif.so: c_src/nx_metal_nif.m c_src/MTLBufferReference.m c_src/MTLDeviceReference.m
	$(CC) $(CFLAGS) -x objective-c $(LDFLAGS) -framework Metal -framework Foundation -o $@ $^

clean:
	$(RM) priv/nx_metal_nif.so
