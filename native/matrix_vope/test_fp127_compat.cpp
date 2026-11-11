#include "fp127.h"
#include "fp127x2.h"
#include "vole/vole_f2k.h"
#include "vole/cvole.h"
// Compile-only coverage of the field APIs used by both VOLE backends.
void instantiate(emp::NetIO** io, FP127* key, FP127x2* value_mac) {
    FP127 delta(7);
    VoleTriple<emp::NetIO, FP127, FP127x2> primal(emp::ALICE,1,io);
    primal.setup<FP127>(delta);
    primal.extend_send(key);
    CVoleFp<emp::NetIO, FP127, FP127x2> committed(emp::ALICE,1,io,cvole_fp_n2to14);
    committed.setup(delta);
    committed.extend_send(key,key);
    CVoleFp<emp::NetIO, FP127, FP127x2> receiver(emp::BOB,1,io,cvole_fp_n2to14);
    receiver.setup_prog(emp::makeBlock(1,2));receiver.setup();
    receiver.extend_recv(value_mac,value_mac);
    RVole<emp::NetIO, FP127, FP127x2> buffered(emp::ALICE,1,io);
    buffered.setup(delta);buffered.vole_send(key,1);
}
