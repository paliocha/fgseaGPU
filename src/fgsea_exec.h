// fgsea_exec.h — execution-policy abstraction.
//
// libstdc++'s parallel std::execution implementation needs Intel TBB at
// link time; macOS's libc++ either doesn't implement parallel policies
// at all (pre-LLVM 17) or has rough edges on the Apple back-end.
//
// To stay portable we route all `for_each` calls through `fgsea::par`,
// which is `std::execution::par_unseq` when the build was configured
// against TBB (`FGSEA_PARALLEL` defined by Makevars on a TBB-enabled
// host) and `std::execution::seq` otherwise. Hot inner loops still go
// through `std::for_each` — they just degrade to sequential on hosts
// without TBB rather than failing to link.

#pragma once

#include <algorithm>
#include <utility>

#if __has_include(<execution>)
#  include <execution>
#endif

namespace fgsea {

// On Apple libc++ the <execution> header exists but the policy tags
// (`std::execution::seq`, `par_unseq`) are not defined. Detect this via
// the standard feature-test macro and fall through to plain `std::for_each`
// (no policy argument) when the policies aren't available — every call
// site then runs sequentially on that host.
#if defined(__cpp_lib_execution) && __cpp_lib_execution >= 201603L
#  ifdef FGSEA_PARALLEL
inline constexpr auto par = std::execution::par_unseq;
#  else
inline constexpr auto par = std::execution::seq;
#  endif

template <class It, class F>
inline void for_each(It first, It last, F f) {
    std::for_each(par, first, last, std::move(f));
}
#else
template <class It, class F>
inline void for_each(It first, It last, F f) {
    std::for_each(first, last, std::move(f));
}
#endif

} // namespace fgsea
