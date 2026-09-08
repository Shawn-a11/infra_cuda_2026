#include "low_precision/quantization.hpp"

#include <stdexcept>

namespace low_precision {

bool cuda_backend_available() { return false; }

QuantizationResult quantize_dequantize_cuda(const Tensor &,
                                            const QuantizationConfig &) {
  throw std::runtime_error(
      "CUDA backend was not built; configure on a machine with nvcc");
}

} // namespace low_precision
