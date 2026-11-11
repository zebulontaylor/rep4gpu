#include "spake2_eq.h"
#include <cstdlib>
#include <iostream>
#ifdef SIV_SPAKE2_EQ_TESTING
#error Production smoke must not expose deterministic test interfaces
#endif
int main() {
 using namespace siv::spake2;
 if(!std::getenv("EMP_TEST_MODE"))return 2;
 bool poisoned=false,transport=false;
 Callbacks cb;
 cb.send=[&](auto){transport=true;};
 cb.receive=[&](auto){transport=true;return Bytes{};};
 cb.accepted_output_barrier=[&](auto){transport=true;};
 cb.poison=[&]{poisoned=true;};
 try{check_private_equal(Role::A,Context{{1},{2},{3},{4}},Bytes{5},cb);}catch(const EqualityFailure&){if(poisoned&&!transport){std::cout<<"production EMP_TEST_MODE rejection: PASS\n";return 0;}}
 return 1;
}
