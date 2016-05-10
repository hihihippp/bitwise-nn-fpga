CC=gcc
CFLAGS=-I. -std=c99 -O2
DEPS = forward.h
OBJ = forward_test.o forward.o 

%.o: %.c $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

forward_test: $(OBJ)
	gcc -o $@ $^ $(CFLAGS)

