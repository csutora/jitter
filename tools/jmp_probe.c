// replicates luajit's bytecode jump-patch arithmetic (lj_parse.c) so a wrong answer names the
// operation instead of surfacing as "control structure too long" three layers up. every input is
// volatile so nothing here can be constant-folded away at -O2.
#include <stdio.h>
#include <stdint.h>
#include <stddef.h>

typedef uint32_t BCPos;
typedef uint32_t BCIns;

#define NO_JMP    (~(BCPos)0)
#define BCBIAS_J  0x8000
#define BCMAX_D   0xffff

#define bc_d(i)   ((i) >> 16)
#define bc_j(i)   ((ptrdiff_t)bc_d(i) - BCBIAS_J)
#define BCINS_AJ(o, a, j) \
  ((BCIns)(o) | ((BCIns)(a) << 8) | ((BCIns)((j) + BCBIAS_J) << 16))

static int fails = 0;

static void ck(const char *what, uint64_t got, uint64_t want) {
    if (got != want) { fails++; printf("FAIL %-34s got 0x%llx want 0x%llx\n", what,
                                       (unsigned long long)got, (unsigned long long)want); }
    else printf("ok   %-34s 0x%llx\n", what, (unsigned long long)got);
}

static BCIns bcbase[64];

// the list walk that ends a chain of pending jumps. if the NO_JMP test is wrong this runs off
// the end of bcbase, which is exactly the observed failure.
static BCPos jmp_next(BCPos pc) {
    ptrdiff_t delta = bc_j(bcbase[pc]);
    if ((BCPos)delta == NO_JMP) return NO_JMP;
    return (BCPos)(((ptrdiff_t)pc + 1) + delta);
}

int main(void) {
    volatile BCPos vno = NO_JMP;
    volatile uint32_t vone = 1;
    volatile BCPos vpc = 3, vdest = 7;

    // the loop guard itself: `while (list != NO_JMP)`
    BCPos list = vno;
    ck("(list != NO_JMP) when equal", list != NO_JMP, 0);
    list = 5;
    ck("(list != NO_JMP) when not", list != NO_JMP, 1);

    // the wrap that makes it: NO_JMP + BCBIAS_J must truncate to 32 bits
    ck("NO_JMP + BCBIAS_J", (BCIns)(vno + BCBIAS_J), 0x7fff);
    ck("(uint32)(0xffffffff + 1)", (uint32_t)(vno + vone), 0);

    // encoding a not-yet-known jump target, then reading it back
    BCIns ins = BCINS_AJ(0x54, 0, vno);
    ck("BCINS_AJ(.., NO_JMP)", ins, 0x7fff0054u);
    ck("bc_d of it", bc_d(ins), 0x7fff);
    ck("bc_j of it", (uint64_t)(int64_t)bc_j(ins), (uint64_t)(int64_t)-1);
    ck("(BCPos)bc_j == NO_JMP", (BCPos)bc_j(ins) == NO_JMP, 1);

    bcbase[3] = ins;
    ck("jmp_next(3) terminates", jmp_next(vpc), NO_JMP);

    bcbase[3] = BCINS_AJ(0x54, 0, (BCPos)2);
    ck("jmp_next(3) follows delta 2", jmp_next(vpc), 6);

    // the range check that raises "control structure too long"
    BCPos offset = vdest - (vpc + 1) + BCBIAS_J;
    ck("forward offset", offset, 0x8003);
    ck("forward offset in range", offset > BCMAX_D, 0);

    volatile BCPos back = 1;
    offset = back - (vpc + 1) + BCBIAS_J;
    ck("backward offset", offset, 0x7ffd);
    ck("backward offset in range", offset > BCMAX_D, 0);

    // and the case that must be rejected: a target that was never resolved
    offset = vno - (vpc + 1) + BCBIAS_J;
    ck("unresolved target rejected", offset > BCMAX_D, 1);

    printf(fails ? "\n%d FAILURES\n" : "\nall %d checks passed\n", fails ? fails : 0);
    return fails != 0;
}
