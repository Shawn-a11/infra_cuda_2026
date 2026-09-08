#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace low_precision {

enum class InputType : std::uint8_t {
  kFloat32 = 1,
  kFloat16 = 2,
};

enum class QuantFormat : std::uint8_t {
  kMxfp8 = 1,
  kNvfp4 = 2,
};

enum class ScaleMode : std::uint8_t {
  kTensor = 1,
  kBlock = 2,
};

enum class OutputType : std::uint8_t {
  kFloat16 = 1,
  kBFloat16 = 2,
  kFloat32 = 3,
};

enum class RoundingMode : std::uint8_t {
  kNearest = 1,
  kStochastic = 2,
};

struct Tensor {
  std::int64_t rows = 0;
  std::int64_t cols = 0;
  InputType source_type = InputType::kFloat32;
  std::vector<float> values;
};

struct QuantizationConfig {
  QuantFormat format = QuantFormat::kMxfp8;
  int block_size = 32;
  ScaleMode scale_mode = ScaleMode::kBlock;
  OutputType output_type = OutputType::kFloat16;
  RoundingMode rounding = RoundingMode::kNearest;
  std::string target_gpu = "T4";
  std::uint64_t seed = 1234;
};

struct QuantizedTensor {
  std::int64_t rows = 0;
  std::int64_t cols = 0;
  QuantFormat format = QuantFormat::kMxfp8;
  ScaleMode scale_mode = ScaleMode::kBlock;
  int block_size = 0;
  // MXFP8: 每个元素 1 byte E4M3；NVFP4: 偶数元素在低 4 bit，奇数元素在高 4
  // bit。
  std::vector<std::uint8_t> packed_values;
  // MXFP8 scale 为 E8M0；NVFP4 局部 scale 为 E4M3。
  std::vector<std::uint8_t> scales;
  // MXFP8 固定为 1；NVFP4 按官方两级缩放公式保存 FP32 tensor scale。
  float global_scale = 1.0f;
};

struct ErrorMetrics {
  double max_absolute_error = 0.0;
  double mean_absolute_error = 0.0;
  double mean_squared_error = 0.0;
  double compression_ratio = 0.0;
};

struct QuantizationResult {
  QuantizedTensor quantized;
  std::vector<float> dequantized;
  ErrorMetrics metrics;
  double quantization_ms = 0.0;
  double dequantization_ms = 0.0;
  std::uint64_t quantization_bytes = 0;
  std::uint64_t dequantization_bytes = 0;
  std::string backend;
};

Tensor load_tensor(const std::string &path);
QuantizationConfig load_quantization_config(const std::string &path);
void save_quantized_tensor(const QuantizedTensor &tensor,
                           const std::string &path);
QuantizedTensor load_quantized_tensor(const std::string &path);
void save_dequantized_tensor(const Tensor &source,
                             const std::vector<float> &values,
                             OutputType output_type, const std::string &path);

QuantizationResult quantize_dequantize_cpu(const Tensor &input,
                                           const QuantizationConfig &config);
// CPU/CUDA 公共入口统一调用该校验，避免直接 API 绕过配置文件约束。
void validate_quantization_request(const Tensor &input,
                                   const QuantizationConfig &config);
bool cuda_backend_available();
QuantizationResult quantize_dequantize_cuda(const Tensor &input,
                                            const QuantizationConfig &config);

ErrorMetrics compute_error_metrics(const Tensor &input,
                                   const QuantizedTensor &quantized,
                                   const std::vector<float> &dequantized);

// 暴露标量编码器用于单元测试和 CPU/CUDA 数值口径对照。
std::uint8_t encode_e4m3(float value, RoundingMode mode,
                         std::uint64_t random_bits);
float decode_e4m3(std::uint8_t bits);
std::uint8_t encode_e2m1(float value, RoundingMode mode,
                         std::uint64_t random_bits);
float decode_e2m1(std::uint8_t bits);
std::uint8_t encode_e8m0_scale(float block_amax);
float decode_e8m0_scale(std::uint8_t bits);

std::string to_string(InputType value);
std::string to_string(QuantFormat value);
std::string to_string(ScaleMode value);
std::string to_string(OutputType value);
std::string to_string(RoundingMode value);

} // namespace low_precision
