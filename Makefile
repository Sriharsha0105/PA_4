NVCC := nvcc
CFLAGS := -O2

all: quamsimV1 quamsimV2

quamsimV1: quamsimV1.cu
	$(NVCC) $(CFLAGS) quamsimV1.cu -o quamsimV1

quamsimV2: quamsimV2.cu
	$(NVCC) $(CFLAGS) quamsimV2.cu -o quamsimV2

clean:
	rm -f quamsimV1 quamsimV2
