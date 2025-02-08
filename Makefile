# Compiler
CC = gcc

# Compiler flags
CFLAGS = -Wall -O3

# Source files
SRCS = charge_transfer.c \
       collisional_ionization.c \
       cooling.c \
       cosmic_ray_ionization.c \
       dust_recombination.c \
       molecules.c \
       photoionization_UVB.c \
       recombination.c \
       rtz_main.c

# Object files (replace .c with .o)
OBJS = $(SRCS:.c=.o)

# Executable name
TARGET = rtz_main

# Default rule to build the executable
all: $(TARGET)

# Rule to link the object files into the final executable
$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $(TARGET) $(OBJS)

# Rule to compile each source file into an object file
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# Clean up the compiled files
clean:
	rm -f $(OBJS) $(TARGET)
