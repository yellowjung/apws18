OBJECTS=main.o qdbmp.o timer.o

CFLAGS=-std=gnu99 -O3 -Wall
LDFLAGS=-lm -lrt

all: seq opencl

seq: $(OBJECTS) facegen_seq.o
	$(CC) -o facegen_seq $^ $(LDFLAGS)

opencl: $(OBJECTS) facegen_opencl.o
	$(CC) -o facegen_opencl $^ $(LDFLAGS) -lOpenCL

clean:
	rm -rf facegen_seq facegen_opencl $(OBJECTS) facegen_seq.o facegen_opencl.o
