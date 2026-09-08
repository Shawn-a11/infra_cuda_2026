#include "low_precision/quantization.hpp"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace low_precision {
namespace {

std::string trim(std::string value) {
  const auto first = std::find_if_not(value.begin(), value.end(), [](char ch) {
    return std::isspace(static_cast<unsigned char>(ch));
  });
  const auto last = std::find_if_not(value.rbegin(), value.rend(), [](char ch) {
                      return std::isspace(static_cast<unsigned char>(ch));
                    }).base();
  if (first >= last)
    return {};
  return std::string(first, last);
}

std::unordered_map<std::string, std::string>
parse_key_values(const std::string &path) {
  std::ifstream input(path);
  if (!input)
    throw std::runtime_error("cannot open config: " + path);
  std::unordered_map<std::string, std::string> values;
  std::string line;
  while (std::getline(input, line)) {
    line = line.substr(0, line.find('#'));
    if (trim(line).empty())
      continue;
    const auto separator = line.find('=');
    if (separator == std::string::npos) {
      throw std::runtime_error("invalid config line in " + path);
    }
    std::string key = trim(line.substr(0, separator));
    std::string value = trim(line.substr(separator + 1));
    if (value.size() >= 2 && value.front() == '"' && value.back() == '"') {
      value = value.substr(1, value.size() - 2);
    }
    if (key.empty() || value.empty() || !values.emplace(key, value).second) {
      throw std::runtime_error("invalid or duplicate config key in " + path);
    }
  }
  return values;
}

const std::string &
required(const std::unordered_map<std::string, std::string> &values,
         const std::string &key, const std::string &path) {
  const auto found = values.find(key);
  if (found == values.end()) {
    throw std::runtime_error("missing '" + key + "' in " + path);
  }
  return found->second;
}

} // namespace

QuantizationConfig load_quantization_config(const std::string &path) {
  // 配置解析在进入量化热路径前完成，严格拒绝未知格式和不兼容的标准块大小。
  const auto values = parse_key_values(path);
  QuantizationConfig config;
  const std::string format = required(values, "format", path);
  if (format == "mxfp8") {
    config.format = QuantFormat::kMxfp8;
  } else if (format == "nvfp4") {
    config.format = QuantFormat::kNvfp4;
  } else {
    throw std::runtime_error("format must be mxfp8 or nvfp4");
  }

  config.block_size = std::stoi(required(values, "block_size", path));
  const std::string scale_mode = required(values, "scale_mode", path);
  if (scale_mode == "tensor") {
    config.scale_mode = ScaleMode::kTensor;
  } else if (scale_mode == "block") {
    config.scale_mode = ScaleMode::kBlock;
  } else {
    throw std::runtime_error("scale_mode must be tensor or block");
  }

  const std::string output_type = required(values, "output_type", path);
  if (output_type == "fp16") {
    config.output_type = OutputType::kFloat16;
  } else if (output_type == "bf16") {
    config.output_type = OutputType::kBFloat16;
  } else if (output_type == "fp32") {
    config.output_type = OutputType::kFloat32;
  } else {
    throw std::runtime_error("output_type must be fp16, bf16 or fp32");
  }

  const std::string rounding = required(values, "rounding", path);
  if (rounding == "nearest") {
    config.rounding = RoundingMode::kNearest;
  } else if (rounding == "stochastic") {
    config.rounding = RoundingMode::kStochastic;
  } else {
    throw std::runtime_error("rounding must be nearest or stochastic");
  }

  config.target_gpu = required(values, "target_gpu", path);
  if (const auto found = values.find("seed"); found != values.end()) {
    config.seed = std::stoull(found->second);
  }
  if (config.block_size <= 0) {
    throw std::runtime_error("block_size must be positive");
  }
  if (config.scale_mode == ScaleMode::kBlock) {
    const int required_block = config.format == QuantFormat::kMxfp8 ? 32 : 16;
    if (config.block_size != required_block) {
      throw std::runtime_error(
          "standards-aligned block mode requires block_size=" +
          std::to_string(required_block));
    }
  }
  return config;
}

std::string to_string(InputType value) {
  return value == InputType::kFloat16 ? "fp16" : "fp32";
}

std::string to_string(QuantFormat value) {
  return value == QuantFormat::kNvfp4 ? "nvfp4" : "mxfp8";
}

std::string to_string(ScaleMode value) {
  return value == ScaleMode::kTensor ? "tensor" : "block";
}

std::string to_string(OutputType value) {
  if (value == OutputType::kFloat16)
    return "fp16";
  if (value == OutputType::kBFloat16)
    return "bf16";
  return "fp32";
}

std::string to_string(RoundingMode value) {
  return value == RoundingMode::kStochastic ? "stochastic" : "nearest";
}

} // namespace low_precision
