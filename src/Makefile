CC := gcc
CFLAGS := -O3 -std=c99
LDFLAGS := -lm

TARGETS := main #main_rml

all: $(TARGETS)

main: main.c weights.h input.h 
	$(CC) $(CFLAGS) -o $@ main.c $(LDFLAGS)

# main_rml: main_rml.c weights_rml.h input_rml.h q_weights_rml.h q_input_rml.h
# 	$(CC) $(CFLAGS) -o $@ main_rml.c $(LDFLAGS)

.PHONY: run clean

run: main
	./main #./main_rml
	
clean:
	rm -f $(TARGETS)