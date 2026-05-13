// fsgea_exec.hpp — execution-policy abstraction.
//
// libstdc++'s parallel std::execution implementation needs Intel TBB at
// link time; macOS's libc++ either doesn't implement parallel policies
// at all (pre-LLVM 17) or has rough edges on the Apple back-end; Rtools
// on Windows ships neither TBB nor a working parallel backend.
//
// To stay portable we route all `for_each` calls through `fsgea::par`,
// which is `std::execution::par_unseq` when the build was configured
// against TBB (`FSGEA_PARALLEL` defined by Makevars on a TBB-enabled
// host) and `std::execution::seq` otherwise. Hot inner loops still go
// through `std::for_each` — they just degrade to sequential on hosts
// without TBB rather than failing to link.

#pragma once

#include <execution>

namespace fsgea {

#ifdef FSGEA_PARALLEL
inline constexpr auto par = std::execution::par_unseq;
#else
inline constexpr auto par = std::execution::seq;
#endif

} // namespace fsgea
