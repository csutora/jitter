// print the derived kernel-SAB layout. every region is sized from PROC_TABLE_SLOTS, so this is how
// you see what raising the process limit actually costs before shipping it.
import { KERNEL_SAB_SIZE, PROC_TABLE_SLOTS, MAILBOX_OFFSET, STDIN_RING_OFFSET,
         GLOBAL_STATE_OFFSET, BRIDGE_RESP_OFFSET, PROC_CMDLINE_OFFSET,
         SIG_SENDER_OFFSET, PROC_CRED_OFFSET } from '../src/kernel-ipc.js';
const h = (n: number) => '0x' + n.toString(16);
console.log('slots     ', PROC_TABLE_SLOTS);
console.log('mailbox   ', h(MAILBOX_OFFSET));
console.log('stdin/pipe', h(STDIN_RING_OFFSET));
console.log('global    ', h(GLOBAL_STATE_OFFSET));
console.log('bridge    ', h(BRIDGE_RESP_OFFSET));
console.log('cmdline   ', h(PROC_CMDLINE_OFFSET));
console.log('sigsender ', h(SIG_SENDER_OFFSET));
console.log('cred      ', h(PROC_CRED_OFFSET));
console.log('SAB MiB   ', (KERNEL_SAB_SIZE / 1048576).toFixed(2));
