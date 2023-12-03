#pragma once

#include <concepts>
#include <cstdint>

namespace ws {

constexpr uint32_t INVALID = 0;

// Utility class to make a primitive data type to be "zeroed" when moved out
// Inspired by ZeroOnMove from https://stackoverflow.com/a/64775626/5394043
// Couldn't figure out how to make this templated on val_'s type
// Also copy operations are deleted so that there can't be two objects having the same handle
class GlHandle {
  uint32_t val_;

 public:
  GlHandle()
      : val_{INVALID} {};
  GlHandle(uint32_t val)
      : val_{val} {}

  GlHandle(const GlHandle& other) = delete;
  GlHandle& operator=(const GlHandle& other) = delete;

  GlHandle(GlHandle&& other) noexcept
      : val_{other.val_} {
    other.val_ = INVALID;
  }
  GlHandle& operator=(GlHandle&& other) noexcept {
    if (&val_ == &other.val_)
      return *this;
    val_ = other.val_;
    other.val_ = INVALID;
    return *this;
  }

  // implicit casting from GlHandle to uint32_t
  operator uint32_t() const {
    return val_;
  }

  // I could have an implicit conversion from GlHandle to uint32_t* to be used in `func(uint32_t* arg)` as `func(zom)`
  // But that would be confusing
  //const uint32_t* ptr() const {
  //  return &val_;
  //}

  // Overloading the reference operator. See "rarely overloaded operators" in https://en.cppreference.com/w/cpp/language/operators
  // Now ampersand will return the address of val not this object's address
  uint32_t* operator&() {
    return &val_;
  }
};
}  // namespace ws
