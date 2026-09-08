#include "low_precision/quantization.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace low_precision {
namespace {

constexpr int kThreads = 256;
constexpr float kE4M3Max = 448.0f;
constexpr float kE2M1Max = 6.0f;

void check_cuda(cudaError_t status, const char *expression) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(expression) + ": " +
                             cudaGetErrorString(status));
  }
}

#define LOW_PRECISION_CUDA_CHECK(expression)                                   \
  check_cuda((expression), #expression)

template <typename T> class DeviceBuffer {
public:
  explicit DeviceBuffer(std::size_t count) : count_(count) {
    if (count_ > 0) {
      LOW_PRECISION_CUDA_CHECK(
          cudaMalloc(reinterpret_cast<void **>(&data_), count_ * sizeof(T)));
    }
  }

  ~DeviceBuffer() {
    if (data_ != nullptr)
      cudaFree(data_);
  }

  DeviceBuffer(const DeviceBuffer &) = delete;
  DeviceBuffer &operator=(const DeviceBuffer &) = delete;

  T *get() { return data_; }
  const T *get() const { return data_; }

private:
  T *data_ = nullptr;
  std::size_t count_ = 0;
};

__device__ std::uint64_t splitmix64_device(std::uint64_t value) {
  value += 0x9e3779b97f4a7c15ULL;
  value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
  value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
  return value ^ (value >> 31U);
}

__device__ double uniform01_device(std::uint64_t bits) {
  return static_cast<double>(bits >> 11U) * 0x1.0p-53;
}

__device__ float decode_e4m3_device(std::uint8_t bits) {
  const bool negative = (bits & 0x80U) != 0;
  const std::uint8_t magnitude = bits & 0x7fU;
  const int exponent = (magnitude >> 3U) & 0x0fU;
  const int mantissa = magnitude & 0x07U;
  if (exponent == 15 && mantissa == 7)
    return nanf("");
  const float value = exponent == 0
                          ? ldexpf(static_cast<float>(mantissa), -9)
                          : ldexpf(1.0f + mantissa / 8.0f, exponent - 7);
  return negative ? -value : value;
}

__device__ float decode_e2m1_device(std::uint8_t bits) {
  const bool negative = (bits & 0x08U) != 0;
  const std::uint8_t magnitude = bits & 0x07U;
  const int exponent = (magnitude >> 1U) & 0x03U;
  const int mantissa = magnitude & 0x01U;
  const float value = exponent == 0
                          ? 0.5f * mantissa
                          : ldexpf(1.0f + 0.5f * mantissa, exponent - 1);
  return negative ? -value : value;
}

__device__ std::uint8_t select_code(float value, std::uint8_t lower,
                                    std::uint8_t upper, float lower_value,
                                    float upper_value, int stochastic,
                                    std::uint64_t random_bits) {
  if (stochastic && upper_value > lower_value) {
    const double probability = static_cast<double>(value - lower_value) /
                               static_cast<double>(upper_value - lower_value);
    return uniform01_device(random_bits) < probability ? upper : lower;
  }
  const float lower_distance = value - lower_value;
  const float upper_distance = upper_value - value;
  if (lower_distance < upper_distance)
    return lower;
  if (upper_distance < lower_distance)
    return upper;
  return (lower & 1U) == 0 ? lower : upper;
}

__device__ std::uint8_t encode_e4m3_device(float value, int stochastic,
                                           std::uint64_t random_bits) {
  if (isnan(value))
    return 0x7fU;
  const bool negative = signbit(value);
  const float magnitude = fminf(fabsf(value), kE4M3Max);
  std::uint8_t positive = 0;
  if (magnitude >= kE4M3Max) {
    positive = 0x7eU;
  } else if (magnitude > 0.0f) {
    for (std::uint16_t code = 1; code <= 0x7eU; ++code) {
      const float upper_value =
          decode_e4m3_device(static_cast<std::uint8_t>(code));
      if (upper_value >= magnitude) {
        const auto upper = static_cast<std::uint8_t>(code);
        const auto lower = static_cast<std::uint8_t>(code - 1);
        positive =
            select_code(magnitude, lower, upper, decode_e4m3_device(lower),
                        upper_value, stochastic, random_bits);
        break;
      }
    }
  }
  return static_cast<std::uint8_t>(positive | (negative ? 0x80U : 0));
}

__device__ std::uint8_t encode_e2m1_device(float value, int stochastic,
                                           std::uint64_t random_bits) {
  const bool negative = signbit(value);
  const float magnitude = fminf(fabsf(value), kE2M1Max);
  std::uint8_t positive = 0;
  if (magnitude >= kE2M1Max) {
    positive = 0x07U;
  } else if (magnitude > 0.0f) {
    for (std::uint8_t code = 1; code <= 0x07U; ++code) {
      const float upper_value = decode_e2m1_device(code);
      if (upper_value >= magnitude) {
        const auto lower = static_cast<std::uint8_t>(code - 1);
        positive =
            select_code(magnitude, lower, code, decode_e2m1_device(lower),
                        upper_value, stochastic, random_bits);
        break;
      }
    }
  }
  return static_cast<std::uint8_t>(positive | (negative ? 0x08U : 0));
}

__device__ std::uint8_t encode_e8m0_device(float amax) {
  if (!(amax > 0.0f))
    return 0;
  int exponent = static_cast<int>(floorf(log2f(amax))) - 8;
  exponent = exponent < -127 ? -127 : (exponent > 127 ? 127 : exponent);
  return static_cast<std::uint8_t>(exponent + 127);
}

__device__ float decode_e8m0_device(std::uint8_t bits) {
  return bits == 0xffU ? nanf("") : ldexpf(1.0f, static_cast<int>(bits) - 127);
}

__global__ void absolute_max_kernel(const float *values, std::size_t count,
                                    unsigned int *maximum_bits) {
  const std::size_t thread =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
  for (std::size_t index = thread; index < count; index += stride) {
    atomicMax(maximum_bits, __float_as_uint(fabsf(values[index])));
  }
}

__global__ void nvfp4_global_scale_kernel(const unsigned int *maximum_bits,
                                          float *global_scale) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    const float maximum = __uint_as_float(*maximum_bits);
    *global_scale = maximum > 0.0f ? maximum / (kE4M3Max * kE2M1Max) : 1.0f;
  }
}

__global__ void compute_scales_kernel(const float *values, std::size_t count,
                                      std::size_t group_size,
                                      std::size_t groups, int format,
                                      const float *global_scale,
                                      std::uint8_t *scales) {
  const std::size_t thread =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
  for (std::size_t group = thread; group < groups; group += stride) {
    const std::size_t begin = group * group_size;
    const std::size_t end =
        begin + group_size < count ? begin + group_size : count;
    float maximum = 0.0f;
    for (std::size_t index = begin; index < end; ++index) {
      maximum = fmaxf(maximum, fabsf(values[index]));
    }
    if (format == static_cast<int>(QuantFormat::kMxfp8)) {
      scales[group] = encode_e8m0_device(maximum);
    } else {
      const float unquantized =
          maximum > 0.0f ? (maximum / kE2M1Max) / *global_scale : 0.0f;
      scales[group] = encode_e4m3_device(unquantized, 0, 0);
    }
  }
}

__global__ void quantize_mxfp8_kernel(const float *values, std::size_t count,
                                      std::size_t group_size,
                                      const std::uint8_t *scales,
                                      int stochastic, std::uint64_t seed,
                                      std::uint8_t *packed) {
  const std::size_t thread =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
  for (std::size_t index = thread; index < count; index += stride) {
    const float scale = decode_e8m0_device(scales[index / group_size]);
    packed[index] = encode_e4m3_device(values[index] / scale, stochastic,
                                       splitmix64_device(seed ^ index));
  }
}

__global__ void quantize_nvfp4_kernel(const float *values, std::size_t count,
                                      std::size_t group_size,
                                      const std::uint8_t *scales,
                                      const float *global_scale, int stochastic,
                                      std::uint64_t seed,
                                      std::uint8_t *packed) {
  const std::size_t thread =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
  const std::size_t bytes = (count + 1) / 2;
  for (std::size_t byte = thread; byte < bytes; byte += stride) {
    const std::size_t first = byte * 2;
    std::uint8_t result = 0;
    for (int lane = 0; lane < 2; ++lane) {
      const std::size_t index = first + static_cast<std::size_t>(lane);
      if (index >= count)
        continue;
      const float combined =
          *global_scale * decode_e4m3_device(scales[index / group_size]);
      const float normalized =
          combined > 0.0f ? values[index] / combined : 0.0f;
      const std::uint8_t code = encode_e2m1_device(
          normalized, stochastic, splitmix64_device(seed ^ index));
      result |= static_cast<std::uint8_t>(code << (lane * 4));
    }
    packed[byte] = result;
  }
}

__global__ void dequantize_kernel(const std::uint8_t *packed, std::size_t count,
                                  std::size_t group_size,
                                  const std::uint8_t *scales, int format,
                                  const float *global_scale, float *output) {
  const std::size_t thread =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
  for (std::size_t index = thread; index < count; index += stride) {
    const std::size_t group = index / group_size;
    if (format == static_cast<int>(QuantFormat::kMxfp8)) {
      output[index] =
          decode_e4m3_device(packed[index]) * decode_e8m0_device(scales[group]);
    } else {
      const std::uint8_t byte = packed[index / 2];
      const std::uint8_t code = (index & 1U) == 0 ? byte & 0x0fU : byte >> 4U;
      output[index] = decode_e2m1_device(code) *
                      decode_e4m3_device(scales[group]) * *global_scale;
    }
  }
}

int launch_blocks(std::size_t work) {
  return std::max(
      1, std::min(65535, static_cast<int>((work + kThreads - 1) / kThreads)));
}

} // namespace

bool cuda_backend_available() {
  int count = 0;
  const cudaError_t status = cudaGetDeviceCount(&count);
  if (status != cudaSuccess) {
    cudaGetLastError();
    return false;
  }
  return count > 0;
}

QuantizationResult quantize_dequantize_cuda(const Tensor &input,
                                            const QuantizationConfig &config) {
  // 初版 CUDA 路径使用通用 CUDA 指令完成缩放、编码、packed store 与解包，
  // 不依赖 Hopper/Blackwell 原生 FP8/FP4 Tensor Core。
  validate_quantization_request(input, config);
  if (!cuda_backend_available()) {
    throw std::runtime_error("no CUDA device is available");
  }

  const std::size_t count = input.values.size();
  const std::size_t group_size =
      config.scale_mode == ScaleMode::kTensor
          ? count
          : static_cast<std::size_t>(config.block_size);
  const std::size_t groups = (count + group_size - 1) / group_size;
  const std::size_t packed_count =
      config.format == QuantFormat::kNvfp4 ? (count + 1) / 2 : count;
  DeviceBuffer<float> device_input(count);
  DeviceBuffer<std::uint8_t> device_scales(groups);
  DeviceBuffer<std::uint8_t> device_packed(packed_count);
  DeviceBuffer<float> device_output(count);
  DeviceBuffer<unsigned int> device_amax(1);
  DeviceBuffer<float> device_global_scale(1);
  LOW_PRECISION_CUDA_CHECK(cudaMemcpy(device_input.get(), input.values.data(),
                                      count * sizeof(float),
                                      cudaMemcpyHostToDevice));
  LOW_PRECISION_CUDA_CHECK(
      cudaMemset(device_amax.get(), 0, sizeof(unsigned int)));
  const float one = 1.0f;
  LOW_PRECISION_CUDA_CHECK(cudaMemcpy(device_global_scale.get(), &one,
                                      sizeof(float), cudaMemcpyHostToDevice));

  cudaEvent_t start = nullptr;
  cudaEvent_t quantized = nullptr;
  cudaEvent_t finished = nullptr;
  LOW_PRECISION_CUDA_CHECK(cudaEventCreate(&start));
  LOW_PRECISION_CUDA_CHECK(cudaEventCreate(&quantized));
  LOW_PRECISION_CUDA_CHECK(cudaEventCreate(&finished));
  LOW_PRECISION_CUDA_CHECK(cudaEventRecord(start));

  if (config.format == QuantFormat::kNvfp4) {
    absolute_max_kernel<<<launch_blocks(count), kThreads>>>(
        device_input.get(), count, device_amax.get());
    nvfp4_global_scale_kernel<<<1, 1>>>(device_amax.get(),
                                        device_global_scale.get());
  }
  compute_scales_kernel<<<launch_blocks(groups), kThreads>>>(
      device_input.get(), count, group_size, groups,
      static_cast<int>(config.format), device_global_scale.get(),
      device_scales.get());
  const int stochastic = config.rounding == RoundingMode::kStochastic ? 1 : 0;
  if (config.format == QuantFormat::kMxfp8) {
    quantize_mxfp8_kernel<<<launch_blocks(count), kThreads>>>(
        device_input.get(), count, group_size, device_scales.get(), stochastic,
        config.seed, device_packed.get());
  } else {
    quantize_nvfp4_kernel<<<launch_blocks(packed_count), kThreads>>>(
        device_input.get(), count, group_size, device_scales.get(),
        device_global_scale.get(), stochastic, config.seed,
        device_packed.get());
  }
  LOW_PRECISION_CUDA_CHECK(cudaGetLastError());
  LOW_PRECISION_CUDA_CHECK(cudaEventRecord(quantized));

  dequantize_kernel<<<launch_blocks(count), kThreads>>>(
      device_packed.get(), count, group_size, device_scales.get(),
      static_cast<int>(config.format), device_global_scale.get(),
      device_output.get());
  LOW_PRECISION_CUDA_CHECK(cudaGetLastError());
  LOW_PRECISION_CUDA_CHECK(cudaEventRecord(finished));
  LOW_PRECISION_CUDA_CHECK(cudaEventSynchronize(finished));

  float quantization_ms = 0.0f;
  float dequantization_ms = 0.0f;
  LOW_PRECISION_CUDA_CHECK(
      cudaEventElapsedTime(&quantization_ms, start, quantized));
  LOW_PRECISION_CUDA_CHECK(
      cudaEventElapsedTime(&dequantization_ms, quantized, finished));

  QuantizationResult result;
  result.quantized.rows = input.rows;
  result.quantized.cols = input.cols;
  result.quantized.format = config.format;
  result.quantized.scale_mode = config.scale_mode;
  result.quantized.block_size = config.block_size;
  result.quantized.packed_values.resize(packed_count);
  result.quantized.scales.resize(groups);
  result.dequantized.resize(count);
  LOW_PRECISION_CUDA_CHECK(cudaMemcpy(result.quantized.packed_values.data(),
                                      device_packed.get(), packed_count,
                                      cudaMemcpyDeviceToHost));
  LOW_PRECISION_CUDA_CHECK(cudaMemcpy(result.quantized.scales.data(),
                                      device_scales.get(), groups,
                                      cudaMemcpyDeviceToHost));
  LOW_PRECISION_CUDA_CHECK(
      cudaMemcpy(result.dequantized.data(), device_output.get(),
                 count * sizeof(float), cudaMemcpyDeviceToHost));
  LOW_PRECISION_CUDA_CHECK(cudaMemcpy(&result.quantized.global_scale,
                                      device_global_scale.get(), sizeof(float),
                                      cudaMemcpyDeviceToHost));
  LOW_PRECISION_CUDA_CHECK(cudaEventDestroy(finished));
  LOW_PRECISION_CUDA_CHECK(cudaEventDestroy(quantized));
  LOW_PRECISION_CUDA_CHECK(cudaEventDestroy(start));

  result.quantization_ms = quantization_ms;
  result.dequantization_ms = dequantization_ms;
  result.quantization_bytes = count * sizeof(float) + groups + packed_count;
  result.dequantization_bytes = groups + packed_count + count * sizeof(float);
  result.backend = "cuda-software-emulation";
  result.metrics =
      compute_error_metrics(input, result.quantized, result.dequantized);
  return result;
}

} // namespace low_precision
