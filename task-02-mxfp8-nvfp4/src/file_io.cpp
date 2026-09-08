#include "low_precision/quantization.hpp"

#include <array>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <type_traits>

namespace low_precision {
namespace {

constexpr std::array<char, 8> kTensorMagic = {'L', 'P', 'T', 'E',
                                              'N', 'S', '1', '\0'};
constexpr std::array<char, 8> kQuantizedMagic = {'L', 'P', 'Q', 'N',
                                                 'T', '0', '1', '\0'};
constexpr std::array<char, 8> kDequantizedMagic = {'L', 'P', 'D', 'E',
                                                   'Q', '0', '1', '\0'};
constexpr std::uint32_t kVersion = 1;

template <typename T> T read_pod(std::istream &input) {
  static_assert(std::is_trivially_copyable_v<T>);
  T value{};
  input.read(reinterpret_cast<char *>(&value), sizeof(value));
  if (!input)
    throw std::runtime_error("truncated binary file");
  return value;
}

template <typename T> void write_pod(std::ostream &output, T value) {
  static_assert(std::is_trivially_copyable_v<T>);
  output.write(reinterpret_cast<const char *>(&value), sizeof(value));
}

void read_magic(std::istream &input, const std::array<char, 8> &expected) {
  std::array<char, 8> actual{};
  input.read(actual.data(), actual.size());
  if (!input || actual != expected) {
    throw std::runtime_error("invalid binary magic");
  }
}

void write_magic(std::ostream &output, const std::array<char, 8> &magic) {
  output.write(magic.data(), magic.size());
}

std::size_t checked_elements(std::int64_t rows, std::int64_t cols) {
  if (rows <= 0 || cols <= 0) {
    throw std::runtime_error("tensor dimensions must be positive");
  }
  const auto unsigned_rows = static_cast<std::uint64_t>(rows);
  const auto unsigned_cols = static_cast<std::uint64_t>(cols);
  if (unsigned_rows > std::numeric_limits<std::size_t>::max() / unsigned_cols) {
    throw std::runtime_error("tensor element count overflows size_t");
  }
  return static_cast<std::size_t>(unsigned_rows * unsigned_cols);
}

float half_to_float(std::uint16_t bits) {
  const std::uint32_t sign = static_cast<std::uint32_t>(bits & 0x8000U) << 16U;
  std::uint32_t exponent = (bits >> 10U) & 0x1fU;
  std::uint32_t mantissa = bits & 0x03ffU;
  std::uint32_t output = 0;
  if (exponent == 0) {
    if (mantissa == 0) {
      output = sign;
    } else {
      int shift = 0;
      while ((mantissa & 0x0400U) == 0) {
        mantissa <<= 1U;
        ++shift;
      }
      mantissa &= 0x03ffU;
      output = sign | static_cast<std::uint32_t>(113 - shift) << 23U |
               mantissa << 13U;
    }
  } else if (exponent == 0x1fU) {
    output = sign | 0x7f800000U | mantissa << 13U;
  } else {
    output = sign | (exponent + 112U) << 23U | mantissa << 13U;
  }
  float value = 0.0f;
  std::memcpy(&value, &output, sizeof(value));
  return value;
}

std::uint16_t float_to_half(float value) {
  std::uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  const std::uint32_t sign = (bits >> 16U) & 0x8000U;
  const std::uint32_t magnitude = bits & 0x7fffffffU;
  if (magnitude >= 0x7f800000U) {
    return static_cast<std::uint16_t>(sign | 0x7c00U |
                                      (magnitude > 0x7f800000U ? 0x0200U : 0));
  }
  int exponent = static_cast<int>((magnitude >> 23U) & 0xffU) - 127 + 15;
  std::uint32_t mantissa = magnitude & 0x7fffffU;
  if (exponent >= 31)
    return static_cast<std::uint16_t>(sign | 0x7bffU);
  if (exponent <= 0) {
    if (exponent < -10)
      return static_cast<std::uint16_t>(sign);
    mantissa |= 0x800000U;
    const int shift = 14 - exponent;
    std::uint32_t rounded = mantissa >> shift;
    const std::uint32_t remainder = mantissa & ((1U << shift) - 1U);
    const std::uint32_t halfway = 1U << (shift - 1);
    if (remainder > halfway || (remainder == halfway && (rounded & 1U))) {
      ++rounded;
    }
    return static_cast<std::uint16_t>(sign | rounded);
  }
  std::uint32_t rounded = mantissa >> 13U;
  const std::uint32_t remainder = mantissa & 0x1fffU;
  if (remainder > 0x1000U || (remainder == 0x1000U && (rounded & 1U))) {
    if (++rounded == 0x400U) {
      rounded = 0;
      ++exponent;
      if (exponent >= 31)
        return static_cast<std::uint16_t>(sign | 0x7bffU);
    }
  }
  return static_cast<std::uint16_t>(
      sign | static_cast<std::uint32_t>(exponent) << 10U | rounded);
}

std::uint16_t float_to_bfloat16(float value) {
  std::uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  const std::uint32_t lsb = (bits >> 16U) & 1U;
  bits += 0x7fffU + lsb;
  return static_cast<std::uint16_t>(bits >> 16U);
}

void ensure_parent(const std::string &path) {
  const auto parent = std::filesystem::path(path).parent_path();
  if (!parent.empty())
    std::filesystem::create_directories(parent);
}

} // namespace

Tensor load_tensor(const std::string &path) {
  // 输入统一转换为 FP32 内存表示，source_type 保留原始字节宽度用于压缩率计算。
  std::ifstream input(path, std::ios::binary);
  if (!input)
    throw std::runtime_error("cannot open tensor: " + path);
  read_magic(input, kTensorMagic);
  if (read_pod<std::uint32_t>(input) != kVersion) {
    throw std::runtime_error("unsupported tensor version");
  }
  Tensor tensor;
  tensor.rows = read_pod<std::int64_t>(input);
  tensor.cols = read_pod<std::int64_t>(input);
  const auto dtype = read_pod<std::uint8_t>(input);
  std::array<char, 3> reserved{};
  input.read(reserved.data(), reserved.size());
  if (!input || (dtype != static_cast<std::uint8_t>(InputType::kFloat32) &&
                 dtype != static_cast<std::uint8_t>(InputType::kFloat16))) {
    throw std::runtime_error("invalid tensor dtype/header");
  }
  tensor.source_type = static_cast<InputType>(dtype);
  const std::size_t count = checked_elements(tensor.rows, tensor.cols);
  tensor.values.resize(count);
  if (tensor.source_type == InputType::kFloat32) {
    input.read(reinterpret_cast<char *>(tensor.values.data()),
               static_cast<std::streamsize>(count * sizeof(float)));
  } else {
    for (std::size_t index = 0; index < count; ++index) {
      tensor.values[index] = half_to_float(read_pod<std::uint16_t>(input));
    }
  }
  if (!input)
    throw std::runtime_error("truncated tensor payload");
  char trailing = 0;
  if (input.read(&trailing, 1)) {
    throw std::runtime_error("tensor file contains trailing bytes");
  }
  return tensor;
}

void save_quantized_tensor(const QuantizedTensor &tensor,
                           const std::string &path) {
  const std::size_t count = checked_elements(tensor.rows, tensor.cols);
  const std::size_t expected =
      tensor.format == QuantFormat::kNvfp4 ? (count + 1) / 2 : count;
  if (tensor.packed_values.size() != expected || tensor.scales.empty() ||
      tensor.block_size <= 0) {
    throw std::invalid_argument("invalid quantized tensor");
  }
  ensure_parent(path);
  std::ofstream output(path, std::ios::binary);
  if (!output)
    throw std::runtime_error("cannot write quantized tensor");
  write_magic(output, kQuantizedMagic);
  write_pod(output, kVersion);
  write_pod(output, tensor.rows);
  write_pod(output, tensor.cols);
  write_pod(output, static_cast<std::uint8_t>(tensor.format));
  write_pod(output, static_cast<std::uint8_t>(tensor.scale_mode));
  write_pod<std::uint16_t>(output, 0);
  write_pod(output, static_cast<std::uint32_t>(tensor.block_size));
  write_pod(output, static_cast<std::uint64_t>(count));
  write_pod(output, static_cast<std::uint64_t>(tensor.packed_values.size()));
  write_pod(output, static_cast<std::uint64_t>(tensor.scales.size()));
  write_pod(output, tensor.global_scale);
  write_pod<std::uint32_t>(output, 0);
  output.write(reinterpret_cast<const char *>(tensor.packed_values.data()),
               static_cast<std::streamsize>(tensor.packed_values.size()));
  output.write(reinterpret_cast<const char *>(tensor.scales.data()),
               static_cast<std::streamsize>(tensor.scales.size()));
  if (!output)
    throw std::runtime_error("failed to write quantized payload");
}

QuantizedTensor load_quantized_tensor(const std::string &path) {
  std::ifstream input(path, std::ios::binary);
  if (!input)
    throw std::runtime_error("cannot open quantized tensor");
  read_magic(input, kQuantizedMagic);
  if (read_pod<std::uint32_t>(input) != kVersion) {
    throw std::runtime_error("unsupported quantized tensor version");
  }
  QuantizedTensor tensor;
  tensor.rows = read_pod<std::int64_t>(input);
  tensor.cols = read_pod<std::int64_t>(input);
  tensor.format = static_cast<QuantFormat>(read_pod<std::uint8_t>(input));
  tensor.scale_mode = static_cast<ScaleMode>(read_pod<std::uint8_t>(input));
  static_cast<void>(read_pod<std::uint16_t>(input));
  tensor.block_size = static_cast<int>(read_pod<std::uint32_t>(input));
  const auto element_count = read_pod<std::uint64_t>(input);
  const auto packed_count = read_pod<std::uint64_t>(input);
  const auto scale_count = read_pod<std::uint64_t>(input);
  tensor.global_scale = read_pod<float>(input);
  static_cast<void>(read_pod<std::uint32_t>(input));
  const std::size_t expected = checked_elements(tensor.rows, tensor.cols);
  if (element_count != expected ||
      packed_count > std::numeric_limits<std::size_t>::max() ||
      scale_count == 0 ||
      scale_count > std::numeric_limits<std::size_t>::max()) {
    throw std::runtime_error("invalid quantized tensor dimensions");
  }
  const std::size_t expected_packed =
      tensor.format == QuantFormat::kNvfp4 ? (expected + 1) / 2 : expected;
  if (packed_count != expected_packed) {
    throw std::runtime_error("invalid packed payload size");
  }
  tensor.packed_values.resize(static_cast<std::size_t>(packed_count));
  tensor.scales.resize(static_cast<std::size_t>(scale_count));
  input.read(reinterpret_cast<char *>(tensor.packed_values.data()),
             static_cast<std::streamsize>(tensor.packed_values.size()));
  input.read(reinterpret_cast<char *>(tensor.scales.data()),
             static_cast<std::streamsize>(tensor.scales.size()));
  if (!input)
    throw std::runtime_error("truncated quantized payload");
  return tensor;
}

void save_dequantized_tensor(const Tensor &source,
                             const std::vector<float> &values,
                             OutputType output_type, const std::string &path) {
  if (values.size() != checked_elements(source.rows, source.cols)) {
    throw std::invalid_argument("dequantized tensor size mismatch");
  }
  ensure_parent(path);
  std::ofstream output(path, std::ios::binary);
  if (!output)
    throw std::runtime_error("cannot write dequantized tensor");
  write_magic(output, kDequantizedMagic);
  write_pod(output, kVersion);
  write_pod(output, source.rows);
  write_pod(output, source.cols);
  write_pod(output, static_cast<std::uint8_t>(output_type));
  std::array<char, 3> reserved{};
  output.write(reserved.data(), reserved.size());
  if (output_type == OutputType::kFloat32) {
    output.write(reinterpret_cast<const char *>(values.data()),
                 static_cast<std::streamsize>(values.size() * sizeof(float)));
  } else {
    for (const float value : values) {
      write_pod(output, output_type == OutputType::kFloat16
                            ? float_to_half(value)
                            : float_to_bfloat16(value));
    }
  }
  if (!output)
    throw std::runtime_error("failed to write dequantized payload");
}

} // namespace low_precision
