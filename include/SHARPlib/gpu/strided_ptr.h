/**
 * \file
 * \brief Strided pointer wrapper for strided multi-dimensional arrays (e.g. [N_levels, N_profiles])
 * \author Kelton Halbert, Aaron
 * \date 2026-10-13
 */

#ifndef SHARP_GPU_STRIDED_PTR_H
#define SHARP_GPU_STRIDED_PTR_H

#include <cstddef>
#include <type_traits>
#include "SHARPlib/constants.h"

namespace sharp {

/**
 * \brief A lightweight pointer wrapper with stride for indexing 2D contiguous arrays
 *
 * Typically used for [N_levels, N_profiles] level-major arrays where
 * ptr points to (base + profile_idx) and stride is N_profiles.
 */
template <typename T>
struct StridedPtr {
    T* ptr = nullptr;
    std::ptrdiff_t stride = 1;

    SHARP_HD constexpr StridedPtr() noexcept = default;

    SHARP_HD constexpr StridedPtr(T* p, std::ptrdiff_t s = 1) noexcept
        : ptr(p), stride(s) {}

    // Enable conversion from StridedPtr<U> to StridedPtr<T> (e.g. non-const to const)
    template <typename U, typename = std::enable_if_t<std::is_convertible_v<U*, T*>>>
    SHARP_HD constexpr StridedPtr(const StridedPtr<U>& other) noexcept
        : ptr(other.ptr), stride(other.stride) {}

    SHARP_HD SHARP_FORCEINLINE T& operator[](std::ptrdiff_t k) const noexcept {
        return ptr[k * stride];
    }

    SHARP_HD SHARP_FORCEINLINE T& operator*() const noexcept {
        return *ptr;
    }

    SHARP_HD SHARP_FORCEINLINE T* operator->() const noexcept {
        return ptr;
    }

    SHARP_HD SHARP_FORCEINLINE explicit operator bool() const noexcept {
        return ptr != nullptr;
    }

    SHARP_HD SHARP_FORCEINLINE StridedPtr operator+(std::ptrdiff_t k) const noexcept {
        return StridedPtr(ptr + k * stride, stride);
    }

    SHARP_HD SHARP_FORCEINLINE StridedPtr& operator+=(std::ptrdiff_t k) noexcept {
        ptr += k * stride;
        return *this;
    }
};

}  // namespace sharp

#endif  // SHARP_GPU_STRIDED_PTR_H
