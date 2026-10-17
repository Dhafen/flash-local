#pragma once
#include <charconv>
#include <cerrno>
#include <filesystem>
#include <fcntl.h>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <unistd.h>

// /dev/fd/N is a capability marker, never a pathname to reopen or stat.
// iOS can deny filesystem traversal of /dev/fd even when fstat/pread on N work.
inline int flash_local_verified_descriptor(const std::string & path) {
    if (path.rfind("/dev/fd/", 0) != 0) return -1;
    int descriptor = -1;
    const char * first = path.data() + 8, * last = path.data() + path.size();
    const auto parsed = std::from_chars(first, last, descriptor);
    if (parsed.ec != std::errc{} || parsed.ptr != last || descriptor < 0)
        throw std::runtime_error("Invalid verified expert descriptor");
    return descriptor;
}

inline uintmax_t flash_local_expert_file_size(const std::string & path, std::error_code & error) {
    const int descriptor = flash_local_verified_descriptor(path);
    if (descriptor < 0) return std::filesystem::file_size(path, error);
    struct stat status{};
    if (fstat(descriptor, &status) != 0) {
        error = std::error_code(errno, std::generic_category()); return uintmax_t(-1);
    }
    if (!S_ISREG(status.st_mode) || status.st_size <= 0) {
        error = std::make_error_code(std::errc::invalid_argument); return uintmax_t(-1);
    }
    error.clear(); return static_cast<uintmax_t>(status.st_size);
}

inline int flash_local_open_experts(const std::string & path) {
    const int descriptor = flash_local_verified_descriptor(path);
    return descriptor < 0 ? open(path.c_str(), O_RDONLY | O_CLOEXEC)
                          : fcntl(descriptor, F_DUPFD_CLOEXEC, 0);
}
