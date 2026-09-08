#include "low_precision/quantization.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace low_precision {
namespace {

constexpr float kE4M3Max = 448.0f;
constexpr float kE2M1Max = 6.0f;

std::uint64_t splitmix64(std::uint64_t value) {
  value += 0x9e3779b97f4a7c15ULL;
  value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
  value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
  return value ^ (value >> 31U);
}

double uniform01(std::uint64_t bits) {
  return static_cast<double>(bits >> 11U) * 0x1.0p-53;
}

template <typename Decode>
std::uint8_t encode_positive(float value, std::uint8_t maximum_code,
                             RoundingMode mode, std::uint64_t random_bits,
                             Decode decode) {
  if (!(value > 0.0f))
    return 0;
  if (value >= decode(maximum_code))
    return maximum_code;
  std::uint8_t lower = 0;
  std::uint8_t upper = 1;
  for (std::uint16_t code = 1; code <= maximum_code; ++code) {
    const float candidate = decode(static_cast<std::uint8_t>(code));
    if (candidate >= value) {
      upper = static_cast<std::uint8_t>(code);
      lower = static_cast<std::uint8_t>(code - 1);
      break;
    }
  }
  const float lower_value = decode(lower);
  const float upper_value = decode(upper);
  if (mode == RoundingMode::kStochastic && upper_value > lower_value) {
    const double probability =
        static_cast<double>(value - lower_value) / (upper_value - lower_value);
    return uniform01(random_bits) < probability ? upper : lower;
  }
  const float lower_distance = value - lower_value;
  const float upper_distance = upper_value - value;
  if (lower_distance < upper_distance)
    return lower;
  if (upper_distance < lower_distance)
    return upper;
  // IEEE round-to-nearest-ties-to-even：选择有效尾数最低位为 0 的编码。
  return (lower & 1U) == 0 ? lower : upper;
}

std::size_t group_size(const Tensor &input, const QuantizationConfig &config) {
  return config.scale_mode == ScaleMode::kTensor
             ? input.values.size()
             : static_cast<std::size_t>(config.block_size);
}

std::size_t scale_count(const Tensor &input, const QuantizationConfig &config) {
  const std::size_t size = group_size(input, config);
  return (input.values.size() + size - 1) / size;
}

float group_amax(const std::vector<float> &values, std::size_t begin,
                 std::size_t end) {
  float maximum = 0.0f;
  for (std::size_t index = begin; index < end; ++index) {
    maximum = std::max(maximum, std::abs(values[index]));
  }
  return maximum;
}

void quantize_mxfp8(const Tensor &input, const QuantizationConfig &config,
                    QuantizedTensor &output) {
  // OCP MXFP8：每组共享 E8M0 幂次 scale，元素使用饱和 E4M3 编码。
  const std::size_t size = group_size(input, config);
  output.scales.resize(scale_count(input, config));
  output.packed_values.resize(input.values.size());
  for (std::size_t group = 0; group < output.scales.size(); ++group) {
    const std::size_t begin = group * size;
    const std::size_t end = std::min(begin + size, input.values.size());
    output.scales[group] =
        encode_e8m0_scale(group_amax(input.values, begin, end));
    const float scale = decode_e8m0_scale(output.scales[group]);
    for (std::size_t index = begin; index < end; ++index) {
      const std::uint64_t random = splitmix64(config.seed ^ index);
      output.packed_values[index] =
          encode_e4m3(input.values[index] / scale, config.rounding, random);
    }
  }
}

void quantize_nvfp4(const Tensor &input, const QuantizationConfig &config,
                    QuantizedTensor &output) {
  // NVIDIA NVFP4：FP32 tensor scale × E4M3 local scale × packed E2M1 value。
  const float global_amax = group_amax(input.values, 0, input.values.size());
  output.global_scale =
      global_amax > 0.0f ? global_amax / (kE4M3Max * kE2M1Max) : 1.0f;
  const std::size_t size = group_size(input, config);
  output.scales.resize(scale_count(input, config));
  output.packed_values.assign((input.values.size() + 1) / 2, 0);
  for (std::size_t group = 0; group < output.scales.size(); ++group) {
    const std::size_t begin = group * size;
    const std::size_t end = std::min(begin + size, input.values.size());
    const float amax = group_amax(input.values, begin, end);
    const float unquantized_scale =
        amax > 0.0f ? (amax / kE2M1Max) / output.global_scale : 0.0f;
    output.scales[group] =
        encode_e4m3(unquantized_scale, RoundingMode::kNearest, 0);
    const float local_scale = decode_e4m3(output.scales[group]);
    const float combined_scale = output.global_scale * local_scale;
    for (std::size_t index = begin; index < end; ++index) {
      const float normalized =
          combined_scale > 0.0f ? input.values[index] / combined_scale : 0.0f;
      const std::uint8_t code = encode_e2m1(normalized, config.rounding,
                                            splitmix64(config.seed ^ index));
      const std::size_t byte = index / 2;
      if ((index & 1U) == 0) {
        output.packed_values[byte] = code & 0x0fU;
      } else {
        output.packed_values[byte] |= static_cast<std::uint8_t>(code << 4U);
      }
    }
  }
}

std::vector<float> dequantize(const QuantizedTensor &input) {
  // 解包路径只依赖落盘字段，验证量化文件确实包含完整恢复信息。
  const std::size_t count = static_cast<std::size_t>(input.rows) * input.cols;
  const std::size_t size = input.scale_mode == ScaleMode::kTensor
                               ? count
                               : static_cast<std::size_t>(input.block_size);
  std::vector<float> output(count);
  for (std::size_t index = 0; index < count; ++index) {
    const std::size_t group = index / size;
    if (input.format == QuantFormat::kMxfp8) {
      output[index] = decode_e4m3(input.packed_values[index]) *
                      decode_e8m0_scale(input.scales[group]);
    } else {
      const std::uint8_t packed = input.packed_values[index / 2];
      const std::uint8_t code =
          (index & 1U) == 0 ? packed & 0x0fU : packed >> 4U;
      output[index] = decode_e2m1(code) * decode_e4m3(input.scales[group]) *
                      input.global_scale;
    }
  }
  return output;
}

} // namespace

void validate_quantization_request(const Tensor &input,
                                   const QuantizationConfig &config) {
  // 先验证乘法不会溢出，再统一检查形状、枚举、标准块大小和有限值策略。
  if (input.rows <= 0 || input.cols <= 0 || config.block_size <= 0) {
    throw std::invalid_argument("invalid tensor/config dimensions");
  }
  const auto rows = static_cast<std::size_t>(input.rows);
  const auto cols = static_cast<std::size_t>(input.cols);
  if (rows > std::numeric_limits<std::size_t>::max() / cols ||
      input.values.empty() || input.values.size() != rows * cols) {
    throw std::invalid_argument("invalid tensor/config dimensions");
  }
  if ((config.format != QuantFormat::kMxfp8 &&
       config.format != QuantFormat::kNvfp4) ||
      (config.scale_mode != ScaleMode::kTensor &&
       config.scale_mode != ScaleMode::kBlock) ||
      (config.rounding != RoundingMode::kNearest &&
       config.rounding != RoundingMode::kStochastic)) {
    throw std::invalid_argument("invalid quantization enum value");
  }
  if (config.scale_mode == ScaleMode::kBlock) {
    const int required_block = config.format == QuantFormat::kMxfp8 ? 32 : 16;
    if (config.block_size != required_block) {
      throw std::invalid_argument(
          "standards-aligned block mode has a fixed block size");
    }
  }
  for (const float value : input.values) {
    if (!std::isfinite(value)) {
      throw std::invalid_argument("quantization accepts finite tensors only");
    }
  }
}

float decode_e4m3(std::uint8_t bits) {
  const bool negative = (bits & 0x80U) != 0;
  const std::uint8_t magnitude = bits & 0x7fU;
  const int exponent = (magnitude >> 3U) & 0x0fU;
  const int mantissa = magnitude & 0x07U;
  if (exponent == 15 && mantissa == 7) {
    return std::numeric_limits<float>::quiet_NaN();
  }
  const float value = exponent == 0
                          ? std::ldexp(static_cast<float>(mantissa), -9)
                          : std::ldexp(1.0f + mantissa / 8.0f, exponent - 7);
  return negative ? -value : value;
}

std::uint8_t encode_e4m3(float value, RoundingMode mode,
                         std::uint64_t random_bits) {
  if (std::isnan(value))
    return 0x7fU;
  const bool negative = std::signbit(value);
  const float magnitude = std::min(std::abs(value), kE4M3Max);
  const auto positive =
      encode_positive(magnitude, 0x7eU, mode, random_bits,
                      [](std::uint8_t code) { return decode_e4m3(code); });
  return static_cast<std::uint8_t>(positive | (negative ? 0x80U : 0));
}

float decode_e2m1(std::uint8_t bits) {
  const bool negative = (bits & 0x08U) != 0;
  const std::uint8_t magnitude = bits & 0x07U;
  const int exponent = (magnitude >> 1U) & 0x03U;
  const int mantissa = magnitude & 0x01U;
  const float value = exponent == 0
                          ? 0.5f * mantissa
                          : std::ldexp(1.0f + 0.5f * mantissa, exponent - 1);
  return negative ? -value : value;
}

std::uint8_t encode_e2m1(float value, RoundingMode mode,
                         std::uint64_t random_bits) {
  const bool negative = std::signbit(value);
  const float magnitude = std::min(std::abs(value), kE2M1Max);
  const auto positive =
      encode_positive(magnitude, 0x07U, mode, random_bits,
                      [](std::uint8_t code) { return decode_e2m1(code); });
  return static_cast<std::uint8_t>(positive | (negative ? 0x08U : 0));
}

std::uint8_t encode_e8m0_scale(float block_amax) {
  // OCP 推荐转换：floor(log2(amax)) 除以 E4M3 最大的 2 次幂 2^8。
  if (!(block_amax > 0.0f))
    return 0;
  int exponent = static_cast<int>(std::floor(std::log2(block_amax))) - 8;
  exponent = std::max(-127, std::min(127, exponent));
  return static_cast<std::uint8_t>(exponent + 127);
}

float decode_e8m0_scale(std::uint8_t bits) {
  if (bits == 0xffU)
    return std::numeric_limits<float>::quiet_NaN();
  return std::ldexp(1.0f, static_cast<int>(bits) - 127);
}

ErrorMetrics compute_error_metrics(const Tensor &input,
                                   const QuantizedTensor &quantized,
                                   const std::vector<float> &dequantized) {
  if (dequantized.size() != input.values.size()) {
    throw std::invalid_argument("metric input size mismatch");
  }
  ErrorMetrics metrics;
  for (std::size_t index = 0; index < input.values.size(); ++index) {
    const double error =
        std::abs(static_cast<double>(input.values[index]) - dequantized[index]);
    metrics.max_absolute_error = std::max(metrics.max_absolute_error, error);
    metrics.mean_absolute_error += error;
    metrics.mean_squared_error += error * error;
  }
  metrics.mean_absolute_error /= input.values.size();
  metrics.mean_squared_error /= input.values.size();
  const std::size_t source_bytes =
      input.values.size() * (input.source_type == InputType::kFloat16
                                 ? sizeof(std::uint16_t)
                                 : sizeof(float));
  const std::size_t quantized_bytes =
      quantized.packed_values.size() + quantized.scales.size() + sizeof(float);
  metrics.compression_ratio =
      static_cast<double>(source_bytes) / quantized_bytes;
  return metrics;
}

QuantizationResult quantize_dequantize_cpu(const Tensor &input,
                                           const QuantizationConfig &config) {
  validate_quantization_request(input, config);
  QuantizationResult result;
  result.quantized.rows = input.rows;
  result.quantized.cols = input.cols;
  result.quantized.format = config.format;
  result.quantized.scale_mode = config.scale_mode;
  result.quantized.block_size = config.block_size;
  result.backend = "cpu-reference";

  const auto quantization_start = std::chrono::steady_clock::now();
  if (config.format == QuantFormat::kMxfp8) {
    quantize_mxfp8(input, config, result.quantized);
  } else {
    quantize_nvfp4(input, config, result.quantized);
  }
  const auto quantization_end = std::chrono::steady_clock::now();
  result.dequantized = dequantize(result.quantized);
  const auto dequantization_end = std::chrono::steady_clock::now();
  result.quantization_ms = std::chrono::duration<double, std::milli>(
                               quantization_end - quantization_start)
                               .count();
  result.dequantization_ms = std::chrono::duration<double, std::milli>(
                                 dequantization_end - quantization_end)
                                 .count();
  result.quantization_bytes = input.values.size() * sizeof(float) +
                              result.quantized.scales.size() +
                              result.quantized.packed_values.size();
  result.dequantization_bytes = result.quantized.scales.size() +
                                result.quantized.packed_values.size() +
                                result.dequantized.size() * sizeof(float);
  result.metrics =
      compute_error_metrics(input, result.quantized, result.dequantized);
  return result;
}

} // namespace low_precision
