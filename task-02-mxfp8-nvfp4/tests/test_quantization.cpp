#include "low_precision/quantization.hpp"

#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(bool condition, const std::string &message) {
  if (!condition)
    throw std::runtime_error(message);
}

template <typename T> void write_pod(std::ostream &output, T value) {
  output.write(reinterpret_cast<const char *>(&value), sizeof(value));
}

std::filesystem::path temporary_path(const std::string &suffix) {
  const auto tick = std::chrono::steady_clock::now().time_since_epoch().count();
  return std::filesystem::temp_directory_path() /
         ("low_precision_test_" + std::to_string(tick) + suffix);
}

low_precision::Tensor tensor_from(std::vector<float> values) {
  low_precision::Tensor tensor;
  tensor.rows = 1;
  tensor.cols = static_cast<std::int64_t>(values.size());
  tensor.source_type = low_precision::InputType::kFloat32;
  tensor.values = std::move(values);
  return tensor;
}

void test_scalar_formats() {
  // E2M1 的全部正数编码来自 OCP 表：0, .5, 1, 1.5, 2, 3, 4, 6。
  const std::array<float, 8> e2m1 = {0.0f, 0.5f, 1.0f, 1.5f,
                                     2.0f, 3.0f, 4.0f, 6.0f};
  for (std::uint8_t code = 0; code < e2m1.size(); ++code) {
    require(low_precision::decode_e2m1(code) == e2m1[code],
            "E2M1 decode table is wrong");
    require(low_precision::encode_e2m1(
                e2m1[code], low_precision::RoundingMode::kNearest, 0) == code,
            "E2M1 exact value did not round-trip");
    require(low_precision::decode_e2m1(code | 0x08U) == -e2m1[code],
            "E2M1 sign bit is wrong");
  }
  require(low_precision::encode_e2m1(
              0.75f, low_precision::RoundingMode::kNearest, 0) == 0x02U,
          "E2M1 ties-to-even is wrong");
  require(low_precision::decode_e4m3(0x38U) == 1.0f,
          "E4M3 one encoding is wrong");
  require(low_precision::decode_e4m3(0x7eU) == 448.0f,
          "E4M3 maximum encoding is wrong");
  require(low_precision::decode_e4m3(0x01U) == std::ldexp(1.0f, -9),
          "E4M3 minimum subnormal is wrong");
  require(low_precision::encode_e4m3(
              448.0f, low_precision::RoundingMode::kNearest, 0) == 0x7eU,
          "E4M3 maximum did not round-trip");
  require(low_precision::decode_e8m0_scale(
              low_precision::encode_e8m0_scale(448.0f)) == 1.0f,
          "E8M0 MX scale calculation is wrong");
}

void test_mxfp8_block_and_tensor() {
  std::vector<float> values(64);
  for (std::size_t index = 0; index < values.size(); ++index) {
    values[index] = static_cast<float>(static_cast<int>(index) - 32) / 3.0f;
  }
  const auto tensor = tensor_from(values);
  low_precision::QuantizationConfig config;
  config.format = low_precision::QuantFormat::kMxfp8;
  config.block_size = 32;
  config.scale_mode = low_precision::ScaleMode::kBlock;
  const auto block = low_precision::quantize_dequantize_cpu(tensor, config);
  require(block.quantized.packed_values.size() == 64 &&
              block.quantized.scales.size() == 2 &&
              block.dequantized.size() == 64,
          "MXFP8 block layout is wrong");
  require(block.metrics.max_absolute_error < 0.6,
          "MXFP8 error is unexpectedly large");

  config.scale_mode = low_precision::ScaleMode::kTensor;
  const auto per_tensor =
      low_precision::quantize_dequantize_cpu(tensor, config);
  require(per_tensor.quantized.scales.size() == 1,
          "MXFP8 per-tensor scaling must store one scale");
}

void test_nvfp4_packing_and_hierarchy() {
  const std::array<float, 8> representable = {0.0f, 0.5f, 1.0f, 1.5f,
                                              2.0f, 3.0f, 4.0f, 6.0f};
  std::vector<float> values;
  for (int repeat = 0; repeat < 2; ++repeat) {
    values.insert(values.end(), representable.begin(), representable.end());
  }
  const auto tensor = tensor_from(values);
  low_precision::QuantizationConfig config;
  config.format = low_precision::QuantFormat::kNvfp4;
  config.block_size = 16;
  config.scale_mode = low_precision::ScaleMode::kBlock;
  const auto result = low_precision::quantize_dequantize_cpu(tensor, config);
  require(result.quantized.packed_values.size() == 8,
          "NVFP4 must pack two values per byte");
  require(result.quantized.scales.size() == 1 &&
              std::abs(result.quantized.global_scale - 1.0f / 448.0f) < 1e-8f,
          "NVFP4 hierarchical scales are wrong");
  require(result.metrics.max_absolute_error < 1e-6,
          "representable NVFP4 values exceed FP32 reconstruction tolerance");
}

void test_stochastic_reproducibility() {
  std::vector<float> values(37, 0.75f);
  const auto tensor = tensor_from(values);
  low_precision::QuantizationConfig config;
  config.format = low_precision::QuantFormat::kNvfp4;
  config.block_size = 16;
  config.rounding = low_precision::RoundingMode::kStochastic;
  config.seed = 99;
  const auto first = low_precision::quantize_dequantize_cpu(tensor, config);
  const auto second = low_precision::quantize_dequantize_cpu(tensor, config);
  require(first.quantized.packed_values == second.quantized.packed_values &&
              first.quantized.scales == second.quantized.scales,
          "stochastic rounding must be reproducible for a fixed seed");
}

void test_edge_cases_and_validation() {
  // 零张量、尾块与奇数元素覆盖最容易出现除零、越界和 padding 污染的边界。
  low_precision::QuantizationConfig mxfp8;
  mxfp8.format = low_precision::QuantFormat::kMxfp8;
  mxfp8.block_size = 32;
  const auto partial = low_precision::quantize_dequantize_cpu(
      tensor_from(std::vector<float>(33, 0.0f)), mxfp8);
  require(partial.quantized.scales.size() == 2 &&
              partial.quantized.packed_values.size() == 33 &&
              partial.dequantized == std::vector<float>(33, 0.0f),
          "MXFP8 zero tensor or partial block is wrong");

  low_precision::QuantizationConfig nvfp4;
  nvfp4.format = low_precision::QuantFormat::kNvfp4;
  nvfp4.block_size = 16;
  const auto odd = low_precision::quantize_dequantize_cpu(
      tensor_from(std::vector<float>(17, 0.0f)), nvfp4);
  require(odd.quantized.scales.size() == 2 &&
              odd.quantized.packed_values.size() == 9 &&
              odd.quantized.global_scale == 1.0f &&
              (odd.quantized.packed_values.back() & 0xf0U) == 0,
          "NVFP4 odd-element padding or zero scale is wrong");

  mxfp8.block_size = 31;
  bool rejected_block_size = false;
  try {
    static_cast<void>(low_precision::quantize_dequantize_cpu(
        tensor_from(std::vector<float>(32, 1.0f)), mxfp8));
  } catch (const std::invalid_argument &) {
    rejected_block_size = true;
  }
  require(rejected_block_size,
          "direct API accepted a non-standard MXFP8 block size");

  for (const float invalid : {std::numeric_limits<float>::quiet_NaN(),
                              std::numeric_limits<float>::infinity()}) {
    bool rejected_non_finite = false;
    try {
      static_cast<void>(low_precision::quantize_dequantize_cpu(
          tensor_from({1.0f, invalid}), nvfp4));
    } catch (const std::invalid_argument &) {
      rejected_non_finite = true;
    }
    require(rejected_non_finite, "NaN/Inf tensor was not rejected");
  }
}

void test_binary_io_and_config() {
  // 手工构造 FP16 输入，验证文档二进制 header 与加载转换。
  const auto tensor_path = temporary_path(".tensor.bin");
  {
    std::ofstream output(tensor_path, std::ios::binary);
    const std::array<char, 8> magic = {'L', 'P', 'T', 'E', 'N', 'S', '1', '\0'};
    output.write(magic.data(), magic.size());
    write_pod<std::uint32_t>(output, 1);
    write_pod<std::int64_t>(output, 1);
    write_pod<std::int64_t>(output, 4);
    write_pod<std::uint8_t>(output, 2);
    const std::array<char, 3> reserved{};
    output.write(reserved.data(), reserved.size());
    const std::array<std::uint16_t, 4> values = {0x3c00U, 0xc000U, 0x3800U,
                                                 0x0000U};
    output.write(reinterpret_cast<const char *>(values.data()),
                 values.size() * sizeof(std::uint16_t));
  }
  const auto loaded = low_precision::load_tensor(tensor_path.string());
  std::filesystem::remove(tensor_path);
  require(loaded.source_type == low_precision::InputType::kFloat16 &&
              loaded.values == std::vector<float>({1.0f, -2.0f, 0.5f, 0.0f}),
          "FP16 input loader is wrong");

  const std::string root = LOW_PRECISION_SOURCE_DIR;
  const auto config = low_precision::load_quantization_config(
      root + "/configs/nvfp4_block.txt");
  require(config.format == low_precision::QuantFormat::kNvfp4 &&
              config.block_size == 16 &&
              config.output_type == low_precision::OutputType::kBFloat16,
          "quantization config parser is wrong");

  auto padded = loaded;
  padded.rows = 1;
  padded.cols = 16;
  padded.values.resize(16, 0.0f);
  const auto result = low_precision::quantize_dequantize_cpu(padded, config);
  const auto quantized_path = temporary_path(".quantized.bin");
  low_precision::save_quantized_tensor(result.quantized,
                                       quantized_path.string());
  const auto round_trip =
      low_precision::load_quantized_tensor(quantized_path.string());
  std::filesystem::remove(quantized_path);
  require(round_trip.packed_values == result.quantized.packed_values &&
              round_trip.scales == result.quantized.scales &&
              round_trip.global_scale == result.quantized.global_scale,
          "quantized file round-trip changed payload");
}

#ifdef LOW_PRECISION_HAS_CUDA
void test_cuda_gate() {
  if (!low_precision::cuda_backend_available())
    return;
  // 65 个元素同时触发 MXFP8 尾块和 NVFP4 奇数 packed byte。
  std::vector<float> values(65);
  for (std::size_t index = 0; index < values.size(); ++index) {
    values[index] = std::sin(static_cast<float>(index)) * 12.0f;
  }
  const auto tensor = tensor_from(values);
  for (const auto format : {low_precision::QuantFormat::kMxfp8,
                            low_precision::QuantFormat::kNvfp4}) {
    for (const auto scale_mode : {low_precision::ScaleMode::kBlock,
                                  low_precision::ScaleMode::kTensor}) {
      for (const auto rounding : {low_precision::RoundingMode::kNearest,
                                  low_precision::RoundingMode::kStochastic}) {
        low_precision::QuantizationConfig config;
        config.format = format;
        config.block_size =
            format == low_precision::QuantFormat::kMxfp8 ? 32 : 16;
        config.scale_mode = scale_mode;
        config.rounding = rounding;
        config.seed = 20260825;
        const auto cpu = low_precision::quantize_dequantize_cpu(tensor, config);
        const auto gpu =
            low_precision::quantize_dequantize_cuda(tensor, config);
        require(cpu.quantized.packed_values == gpu.quantized.packed_values &&
                    cpu.quantized.scales == gpu.quantized.scales &&
                    cpu.quantized.global_scale == gpu.quantized.global_scale,
                "CPU/CUDA encoded payload mismatch");
        for (std::size_t index = 0; index < values.size(); ++index) {
          require(std::abs(cpu.dequantized[index] - gpu.dequantized[index]) <
                      1e-6f,
                  "CPU/CUDA dequantized value mismatch");
        }
      }
    }
  }
}
#endif

} // namespace

int main() {
  try {
    test_scalar_formats();
    test_mxfp8_block_and_tensor();
    test_nvfp4_packing_and_hierarchy();
    test_stochastic_reproducibility();
    test_edge_cases_and_validation();
    test_binary_io_and_config();
#ifdef LOW_PRECISION_HAS_CUDA
    test_cuda_gate();
#endif
    std::cout << "all low-precision tests passed\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "test failure: " << error.what() << '\n';
    return 1;
  }
}
