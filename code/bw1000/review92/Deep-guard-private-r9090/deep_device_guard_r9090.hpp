#ifndef DEEP_PRIVATE_DEVICE_GUARD_R9090_H_
#define DEEP_PRIVATE_DEVICE_GUARD_R9090_H_

#include <infini/rt.h>

#include <cstdio>
#include <atomic>
#include <array>
#include <stdexcept>

namespace infini::rt {
inline std::atomic<int> deep_guard_mode_r9090{0};
inline std::array<std::array<std::atomic<uint64_t>,4>,8> deep_guard_counts_r9090{};

// Select the owning resource's device and restore this thread's prior
// selection.
class DeepDeviceGuardR9090 {
 public:
  DeepDeviceGuardR9090(Device::Type target_type, int target_index) noexcept
      : previous_type_(runtime_device_type()), target_index_r9090_(target_index), optimize_r9090_(deep_guard_mode_r9090.load()==1) {
    if (!enabled(target_type, AllDeviceTypes{})) {
      std::fputs("InfiniRT device scope unsupported target device type\n",
                 stderr);
      restored_ = true;
      return;
    }
    const auto previous_status = runtime::GetDevice(&previous_index_);
    restore_index_ = previous_status == runtime::kSuccess;
    warn(previous_status, "read previous device");
    if (restore_index_ && previous_type_ == target_type &&
        previous_index_ == target_index) {
      active_ = true;
      return;
    }
    set_runtime_device_type(target_type);
    const auto status = runtime::SetDevice(target_index);
    active_ = status == runtime::kSuccess;
    warn(status, "select target device");
    if (!active_) restore();
  }

  DeepDeviceGuardR9090(Device::Type target_type, int target_index, bool required)
      : DeepDeviceGuardR9090(target_type, target_index) {
    if (required && !active_)
      throw std::runtime_error("cannot select runtime device");
  }

  DeepDeviceGuardR9090(const DeepDeviceGuardR9090&) = delete;
  DeepDeviceGuardR9090& operator=(const DeepDeviceGuardR9090&) = delete;
  DeepDeviceGuardR9090(DeepDeviceGuardR9090&&) = delete;
  DeepDeviceGuardR9090& operator=(DeepDeviceGuardR9090&&) = delete;
  ~DeepDeviceGuardR9090() noexcept { restore(); }

  bool active() const noexcept { return active_; }

 private:
  template <Device::Type... types>
  static bool enabled(Device::Type type, List<types...>) noexcept {
    return ((type == types && DeviceEnabled<types>::value) || ...);
  }

  static void warn(runtime::Error status, const char* operation) noexcept {
    if (status != runtime::kSuccess)
      std::fprintf(stderr, "InfiniRT device scope %s error=%lld\n", operation,
                   static_cast<long long>(status));
  }
  void restore() noexcept {
    if (restored_) return;
    restored_ = true;
    set_runtime_device_type(previous_type_);
    if (restore_index_) {
      auto &c=deep_guard_counts_r9090.at(target_index_r9090_); ++c[0];
      if(optimize_r9090_) {
        ++c[1]; int current=-1;
        const auto status=runtime::GetDevice(&current);
        if(status==runtime::kSuccess && current==previous_index_) {++c[2]; return;}
      }
      ++c[3]; warn(runtime::SetDevice(previous_index_), "restore previous device");
    }
  }

  int target_index_r9090_;
  bool optimize_r9090_;
  Device::Type previous_type_;
  int previous_index_ = 0;
  bool restore_index_ = false;
  bool active_ = false;
  bool restored_ = false;
};

}  // namespace infini::rt

#endif
