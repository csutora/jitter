CC = aarch64-unknown-linux-musl-gcc
CFLAGS = -static -mgeneral-regs-only -O1 -Wall -I..
SRCDIR = src
BINDIR = bin

SOURCES := $(wildcard $(SRCDIR)/test_*.c)
TARGETS := $(patsubst $(SRCDIR)/%.c,$(BINDIR)/%,$(SOURCES))

all: $(TARGETS)

$(BINDIR)/%: $(SRCDIR)/%.c harness.h
	@mkdir -p $(BINDIR)
	$(CC) $(CFLAGS) -o $@ $<

clean:
	rm -f $(BINDIR)/test_*

.PHONY: all clean
