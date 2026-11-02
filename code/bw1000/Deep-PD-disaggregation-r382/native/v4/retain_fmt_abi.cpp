#include <spdlog/fmt/fmt.h>
// Preserve the parent's emitted weak inline ABI, without replacing its body or
// relaxing the exact original-export check. Same pinned fmt12 header as CPU766.
namespace {
using Args = fmt::basic_format_args<fmt::context>;
using Arg = fmt::basic_format_arg<fmt::context>;
[[gnu::used]] Arg (Args::*retained_fmt_args_get)(int) const = &Args::get;
}
