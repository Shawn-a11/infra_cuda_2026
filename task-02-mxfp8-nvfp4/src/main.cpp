#include "low_precision/quantization.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

struct Arguments {
  std::string input_path;
  std::string config_path;
  std::string backend = "auto";
  std::string quantized_path = "outputs/quantized.bin";
  std::string dequantized_path = "outputs/dequantized.bin";
  std::string metrics_path = "outputs/metrics.log";
  std::string performance_path = "outputs/performance.log";
};

void usage(const char *program) {
  std::cout << "Usage: " << program
            << " --input FILE --config FILE [--backend auto|cpu|cuda]"
               " [--quantized FILE] [--dequantized FILE]"
               " [--metrics FILE] [--performance FILE]\n";
}

Arguments parse_arguments(int argc, char **argv) {
  Arguments arguments;
  for (int index = 1; index < argc; ++index) {
    const std::string flag = argv[index];
    if (flag == "--help" || flag == "-h") {
      usage(argv[0]);
      std::exit(0);
    }
    if (index + 1 >= argc)
      throw std::runtime_error("missing value after " + flag);
    const std::string value = argv[++index];
    if (flag == "--input") {
      arguments.input_path = value;
    } else if (flag == "--config") {
      arguments.config_path = value;
    } else if (flag == "--backend") {
      arguments.backend = value;
    } else if (flag == "--quantized") {
      arguments.quantized_path = value;
    } else if (flag == "--dequantized") {
      arguments.dequantized_path = value;
    } else if (flag == "--metrics") {
      arguments.metrics_path = value;
    } else if (flag == "--performance") {
      arguments.performance_path = value;
    } else {
      throw std::runtime_error("unknown argument: " + flag);
    }
  }
  if (arguments.input_path.empty() || arguments.config_path.empty()) {
    throw std::runtime_error("--input and --config are required");
  }
  if (arguments.backend != "auto" && arguments.backend != "cpu" &&
      arguments.backend != "cuda") {
    throw std::runtime_error("--backend must be auto, cpu or cuda");
  }
  return arguments;
}

void ensure_parent(const std::string &path) {
  const auto parent = std::filesystem::path(path).parent_path();
  if (!parent.empty())
    std::filesystem::create_directories(parent);
}

double bandwidth_gbps(std::uint64_t bytes, double milliseconds) {
  return milliseconds > 0.0
             ? static_cast<double>(bytes) / (milliseconds * 1.0e6)
             : 0.0;
}

} // namespace

int main(int argc, char **argv) {
  try {
    const Arguments arguments = parse_arguments(argc, argv);
    const auto input = low_precision::load_tensor(arguments.input_path);
    const auto config =
        low_precision::load_quantization_config(arguments.config_path);
    const bool cuda_available = low_precision::cuda_backend_available();
    if (arguments.backend == "cuda" && !cuda_available) {
      throw std::runtime_error("CUDA backend requested but unavailable");
    }
    const bool use_cuda = arguments.backend == "cuda" ||
                          (arguments.backend == "auto" && cuda_available);
    const auto wall_start = std::chrono::steady_clock::now();
    const auto result =
        use_cuda ? low_precision::quantize_dequantize_cuda(input, config)
                 : low_precision::quantize_dequantize_cpu(input, config);
    const auto wall_end = std::chrono::steady_clock::now();
    const double wall_ms =
        std::chrono::duration<double, std::milli>(wall_end - wall_start)
            .count();

    // CUDA 运行同时生成同口径 CPU reference，结果文件记录
    // packed/scale/反量化差异。
    const auto cpu_reference =
        use_cuda ? low_precision::quantize_dequantize_cpu(input, config)
                 : result;
    double reference_difference = 0.0;
    for (std::size_t index = 0; index < result.dequantized.size(); ++index) {
      reference_difference =
          std::max(reference_difference,
                   std::abs(static_cast<double>(result.dequantized[index]) -
                            cpu_reference.dequantized[index]));
    }
    const bool packed_match =
        result.quantized.packed_values == cpu_reference.quantized.packed_values;
    const bool scale_match =
        result.quantized.scales == cpu_reference.quantized.scales;

    low_precision::save_quantized_tensor(result.quantized,
                                         arguments.quantized_path);
    low_precision::save_dequantized_tensor(input, result.dequantized,
                                           config.output_type,
                                           arguments.dequantized_path);

    ensure_parent(arguments.metrics_path);
    std::ofstream metrics(arguments.metrics_path);
    if (!metrics)
      throw std::runtime_error("cannot write metrics log");
    metrics << std::setprecision(12)
            << "format=" << low_precision::to_string(config.format) << '\n'
            << "scale_mode=" << low_precision::to_string(config.scale_mode)
            << '\n'
            << "rounding=" << low_precision::to_string(config.rounding) << '\n'
            << "max_absolute_error=" << result.metrics.max_absolute_error
            << '\n'
            << "mean_absolute_error=" << result.metrics.mean_absolute_error
            << '\n'
            << "mean_squared_error=" << result.metrics.mean_squared_error
            << '\n'
            << "compression_ratio=" << result.metrics.compression_ratio << '\n'
            << "cpu_reference_packed_match="
            << (packed_match ? "true" : "false") << '\n'
            << "cpu_reference_scale_match=" << (scale_match ? "true" : "false")
            << '\n'
            << "cpu_reference_max_dequantized_difference="
            << reference_difference << '\n';

    ensure_parent(arguments.performance_path);
    std::ofstream performance(arguments.performance_path);
    if (!performance)
      throw std::runtime_error("cannot write performance log");
    const double kernel_ms = result.quantization_ms + result.dequantization_ms;
    const double cpu_ms =
        cpu_reference.quantization_ms + cpu_reference.dequantization_ms;
    performance
        << std::setprecision(12) << "backend=" << result.backend << '\n'
        << "target_gpu=" << config.target_gpu << '\n'
        << "num_rows=" << input.rows << '\n'
        << "num_cols=" << input.cols << '\n'
        << "num_elements=" << input.values.size() << '\n'
        << "input_type=" << low_precision::to_string(input.source_type) << '\n'
        << "output_type=" << low_precision::to_string(config.output_type)
        << '\n'
        << "format=" << low_precision::to_string(config.format) << '\n'
        << "block_size=" << config.block_size << '\n'
        << "scale_mode=" << low_precision::to_string(config.scale_mode) << '\n'
        << "rounding=" << low_precision::to_string(config.rounding) << '\n'
        << "seed=" << config.seed << '\n'
        << "packed_bytes=" << result.quantized.packed_values.size() << '\n'
        << "scale_bytes=" << result.quantized.scales.size() << '\n'
        << "global_scale=" << result.quantized.global_scale << '\n'
        << "quantization_kernel_ms=" << result.quantization_ms << '\n'
        << "dequantization_kernel_ms=" << result.dequantization_ms << '\n'
        << "kernel_total_ms=" << kernel_ms << '\n'
        << "end_to_end_compute_ms=" << wall_ms << '\n'
        << "quantization_effective_bandwidth_gbps="
        << bandwidth_gbps(result.quantization_bytes, result.quantization_ms)
        << '\n'
        << "dequantization_effective_bandwidth_gbps="
        << bandwidth_gbps(result.dequantization_bytes, result.dequantization_ms)
        << '\n'
        << "cpu_reference_ms=" << cpu_ms << '\n'
        << "speedup_vs_cpu_reference="
        << (kernel_ms > 0.0 ? cpu_ms / kernel_ms : 0.0) << '\n'
        << "timing_scope=quantize_and_dequantize_compute_only\n"
        << "input_conversion_and_file_io_included=false\n";

    std::cout << std::setprecision(6) << "backend=" << result.backend
              << " format=" << low_precision::to_string(config.format)
              << " mae=" << result.metrics.mean_absolute_error
              << " mse=" << result.metrics.mean_squared_error
              << " compression=" << result.metrics.compression_ratio
              << "x total_ms=" << wall_ms << '\n';
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
}
