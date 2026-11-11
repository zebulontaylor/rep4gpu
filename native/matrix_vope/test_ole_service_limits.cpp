#include "ole_service_limits.h"
#include <array>
#include <cassert>

int main() {
    constexpr auto maximum=UINT64_MAX;
    const std::array<std::array<uint64_t,4>,8> accepted{{
        {0,0,0,1},{maximum-1,0,0,1},{0,maximum-508,maximum-1,1},
        {0,maximum-1016,maximum-2,2},{3,3556,7,4096},
        {0,0,0,maximum/508},{maximum-1,maximum-508,maximum-1,1},{1,508,1,1}}};
    const std::array<std::array<uint64_t,4>,6> rejected{{
        {maximum,0,0,1},{0,maximum-507,0,1},{0,0,maximum,1},
        {0,maximum-1015,0,2},{0,0,0,maximum/508+1},{0,0,0,0}}};
    for(const auto& c:accepted)matrix_vope::require_ole_service_capacity(c[0],c[1],c[2],c[3]);
    for(const auto& c:rejected) {
        bool failed=false;
        try {matrix_vope::require_ole_service_capacity(c[0],c[1],c[2],c[3]);}
        catch(const std::runtime_error&){failed=true;}
        assert(failed);
    }
}
