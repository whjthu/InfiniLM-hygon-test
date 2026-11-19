#pragma once

#include <chrono>
#include <cstdio>
#include <stdexcept>

#if defined(__linux__)
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace kimi_stage {

// The diagnostic controller redirects the child's stderr to its private file.
// The regular-file sync keeps completed boundaries visible even after SIGKILL.
inline void mark(const char *name) {
    using Clock = std::chrono::steady_clock;
    static thread_local unsigned sequence = 0;
    const auto now = std::chrono::duration_cast<std::chrono::nanoseconds>(
        Clock::now().time_since_epoch()).count();
    if (std::fprintf(stderr, "kimi-stage native %u %lld %s\n", sequence++,
                     static_cast<long long>(now), name) < 0 || std::fflush(stderr) != 0) {
        throw std::runtime_error("could not write Kimi diagnostic stage");
    }
#if defined(__linux__)
    const int fd = fileno(stderr);
    struct stat status {};
    if (fstat(fd, &status) != 0 || (S_ISREG(status.st_mode) && fdatasync(fd) != 0)) {
        throw std::runtime_error("could not persist Kimi diagnostic stage");
    }
#endif
}

} // namespace kimi_stage
