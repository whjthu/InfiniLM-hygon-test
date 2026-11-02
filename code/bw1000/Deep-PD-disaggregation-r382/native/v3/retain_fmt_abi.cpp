#include <spdlog/fmt/fmt.h>
// Preserve the parent's emitted weak inline ABI, without replacing its body or
// relaxing the exact original-export check. Same pinned fmt12 header as CPU766.
namespace {
[[gnu::used]] auto retained_fmt_args_get = &fmt::basic_format_args<fmt::context>::get;
}
