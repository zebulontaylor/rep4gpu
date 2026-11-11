#pragma once
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace matrix_vope {
// A completed request must leave representable counters for its next close or
// request. Check before emitting a header or consuming any OT results.
inline void require_ole_service_capacity(uint64_t sequence,uint64_t rot_position,
                                         uint64_t completed_calls,uint64_t calls) {
    constexpr auto maximum=std::numeric_limits<uint64_t>::max();
    if(!calls || sequence==maximum || calls>(maximum-rot_position)/508 ||
       calls>maximum-completed_calls)
        throw std::runtime_error("OLE service counter capacity exhausted");
}
}
