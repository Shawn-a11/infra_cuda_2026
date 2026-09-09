#include "vector_search/search.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace vector_search {
namespace {

float score_vector(const float* left, const float* right, int dim,
                   Metric metric) {
  if (metric == Metric::kL2) {
    float distance = 0.0f;
    for (int component = 0; component < dim; ++component) {
      const float difference = left[component] - right[component];
      distance += difference * difference;
    }
    return distance;
  }
  float dot = 0.0f;
  float left_norm = 0.0f;
  float right_norm = 0.0f;
  for (int component = 0; component < dim; ++component) {
    dot += left[component] * right[component];
    if (metric == Metric::kCosine) {
      left_norm += left[component] * left[component];
      right_norm += right[component] * right[component];
    }
  }
  if (metric == Metric::kCosine) {
    const float denominator = std::sqrt(left_norm * right_norm);
    return denominator > 0.0f ? dot / denominator : 0.0f;
  }
  return dot;
}

bool better(float left, float right, Metric metric) {
  return metric == Metric::kL2 ? left < right : left > right;
}

std::vector<int> assign_nearest_cpu(const std::vector<float>& vectors,
                                    std::int64_t count, int dim,
                                    const std::vector<float>& centers,
                                    int nlist, Metric metric) {
  // Lloyd 训练和最终建库共用该分配器：逐向量扫描所有中心，返回最优 list id。
  if (count <= 0 || dim <= 0 || nlist <= 0 ||
      vectors.size() != static_cast<std::size_t>(count) * dim ||
      centers.size() != static_cast<std::size_t>(nlist) * dim) {
    throw std::invalid_argument("invalid IVF assignment dimensions");
  }
  std::vector<int> assignments(static_cast<std::size_t>(count));
  for (std::int64_t vector_id = 0; vector_id < count; ++vector_id) {
    const float* vector = vectors.data() + vector_id * dim;
    int best_list = 0;
    float best_score = score_vector(vector, centers.data(), dim, metric);
    for (int list = 1; list < nlist; ++list) {
      const float score =
          score_vector(vector, centers.data() + list * dim, dim, metric);
      if (better(score, best_score, metric)) {
        best_score = score;
        best_list = list;
      }
    }
    assignments[static_cast<std::size_t>(vector_id)] = best_list;
  }
  return assignments;
}

std::vector<int> assign_nearest(const std::vector<float>& vectors,
                                std::int64_t count, int dim,
                                const std::vector<float>& centers, int nlist,
                                Metric metric, bool use_cuda) {
  // 训练器通过一个入口切换 CPU/CUDA assignment，保证两条路径共享 k-means 主流程。
  return use_cuda
             ? assign_nearest_cuda(vectors, count, dim, centers, nlist, metric)
             : assign_nearest_cpu(vectors, count, dim, centers, nlist, metric);
}

void normalize(float* vector, int dim) {
  // Cosine k-means 的中心保持单位范数；零向量保持为零以匹配搜索度量定义。
  double square_sum = 0.0;
  for (int component = 0; component < dim; ++component) {
    square_sum += static_cast<double>(vector[component]) * vector[component];
  }
  const double norm = std::sqrt(square_sum);
  if (norm > 0.0) {
    for (int component = 0; component < dim; ++component) {
      vector[component] = static_cast<float>(vector[component] / norm);
    }
  }
}

std::vector<float> make_training_set(const VectorDatabase& database,
                                     std::int64_t training_count,
                                     std::uint64_t seed) {
  // 采用带 seed 偏移的等距确定性采样，避免复制全库，同时保证索引可复现。
  std::vector<float> training(static_cast<std::size_t>(training_count) *
                              database.dim);
  const std::uint64_t offset = seed % static_cast<std::uint64_t>(database.count);
  for (std::int64_t sample = 0; sample < training_count; ++sample) {
    const auto base_id = static_cast<std::int64_t>(
        (static_cast<std::uint64_t>(sample) * database.count / training_count +
         offset) %
        static_cast<std::uint64_t>(database.count));
    std::copy_n(database.values.data() + base_id * database.dim, database.dim,
                training.data() + sample * database.dim);
  }
  return training;
}

std::vector<int> ranked_centers(const float* query, const IvfIndex& index) {
  // Coarse search 先对全部中心排序；相同中心分数按 list id 决定顺序。
  std::vector<int> lists(static_cast<std::size_t>(index.nlist));
  std::iota(lists.begin(), lists.end(), 0);
  std::vector<float> scores(static_cast<std::size_t>(index.nlist));
  for (int list = 0; list < index.nlist; ++list) {
    scores[static_cast<std::size_t>(list)] =
        score_vector(query, index.centers.data() + list * index.dim, index.dim,
                     index.metric);
  }
  std::sort(lists.begin(), lists.end(), [&](int left, int right) {
    const float left_score = scores[static_cast<std::size_t>(left)];
    const float right_score = scores[static_cast<std::size_t>(right)];
    if (left_score != right_score) {
      return better(left_score, right_score, index.metric);
    }
    return left < right;
  });
  return lists;
}

void validate_index_compatibility(const VectorDatabase& database,
                                  const IvfIndex& index,
                                  const SearchConfig& config) {
  // 在候选生成前验证数据库、索引和 nprobe 的结构兼容性。
  if (index.num_vectors != database.count || index.dim != database.dim ||
      index.metric != database.metric || index.nlist <= 0 ||
      index.offsets.size() != static_cast<std::size_t>(index.nlist + 1) ||
      index.ids.size() != static_cast<std::size_t>(database.count) ||
      config.nprobe <= 0 || config.nprobe > index.nlist) {
    throw std::invalid_argument("IVF index is incompatible with database/config");
  }
}

}  // namespace

IvfIndex build_ivf_index(const VectorDatabase& database,
                         const SearchConfig& config, bool use_cuda) {
  // IVF-Flat 构建分为训练集采样、Lloyd 迭代、全库最终分配和 CSR 风格
  // offsets/ids 生成。CUDA 当前只负责 assignment，质心更新仍在 CPU。
  if (database.count <= 0 || database.dim <= 0 || config.nlist <= 0 ||
      config.nlist > database.count || config.training_samples < config.nlist) {
    throw std::invalid_argument("invalid database or IVF training dimensions");
  }
  const std::int64_t training_count =
      std::min(database.count, config.training_samples);
  std::vector<float> training =
      make_training_set(database, training_count, config.seed);

  IvfIndex index;
  index.num_vectors = database.count;
  index.dim = database.dim;
  index.nlist = config.nlist;
  index.metric = database.metric;
  index.centers.resize(static_cast<std::size_t>(config.nlist) * database.dim);
  for (int list = 0; list < config.nlist; ++list) {
    const std::int64_t sample =
        static_cast<std::int64_t>(list) * training_count / config.nlist;
    std::copy_n(training.data() + sample * database.dim, database.dim,
                index.centers.data() + list * database.dim);
    if (database.metric == Metric::kCosine) {
      normalize(index.centers.data() + list * database.dim, database.dim);
    }
  }

  for (int iteration = 0; iteration < config.kmeans_iterations; ++iteration) {
    const auto assignments =
        assign_nearest(training, training_count, database.dim, index.centers,
                       config.nlist, database.metric, use_cuda);
    std::vector<double> sums(index.centers.size(), 0.0);
    std::vector<std::uint64_t> counts(static_cast<std::size_t>(config.nlist), 0);
    for (std::int64_t sample = 0; sample < training_count; ++sample) {
      const int list = assignments[static_cast<std::size_t>(sample)];
      ++counts[static_cast<std::size_t>(list)];
      for (int component = 0; component < database.dim; ++component) {
        sums[static_cast<std::size_t>(list) * database.dim + component] +=
            training[static_cast<std::size_t>(sample) * database.dim +
                     component];
      }
    }
    for (int list = 0; list < config.nlist; ++list) {
      float* center = index.centers.data() + list * database.dim;
      if (counts[static_cast<std::size_t>(list)] == 0) {
        const std::int64_t replacement =
            (static_cast<std::int64_t>(iteration + 1) * config.nlist + list) %
            training_count;
        std::copy_n(training.data() + replacement * database.dim, database.dim,
                    center);
      } else {
        const double inverse =
            1.0 / static_cast<double>(counts[static_cast<std::size_t>(list)]);
        for (int component = 0; component < database.dim; ++component) {
          center[component] = static_cast<float>(
              sums[static_cast<std::size_t>(list) * database.dim + component] *
              inverse);
        }
      }
      if (database.metric == Metric::kCosine) normalize(center, database.dim);
    }
  }

  const auto assignments =
      assign_nearest(database.values, database.count, database.dim,
                     index.centers, config.nlist, database.metric, use_cuda);
  index.offsets.assign(static_cast<std::size_t>(config.nlist + 1), 0);
  for (const int list : assignments) {
    ++index.offsets[static_cast<std::size_t>(list + 1)];
  }
  std::partial_sum(index.offsets.begin(), index.offsets.end(),
                   index.offsets.begin());
  std::vector<std::uint64_t> cursors = index.offsets;
  index.ids.resize(static_cast<std::size_t>(database.count));
  for (std::int64_t vector_id = 0; vector_id < database.count; ++vector_id) {
    const int list = assignments[static_cast<std::size_t>(vector_id)];
    index.ids[static_cast<std::size_t>(cursors[static_cast<std::size_t>(list)]++)] =
        vector_id;
  }
  return index;
}

SearchResults search_ivf(const VectorDatabase& database,
                         const QuerySet& queries, const IvfIndex& index,
                         const SearchConfig& config, bool use_cuda) {
  // 每个 query 先选择 nprobe 个倒排表，再对候选做精确 rerank；若候选不足 K，
  // 会继续探测额外 list，以保证结果缓冲区尽可能填满。
  validate_index_compatibility(database, index, config);
  if (queries.count <= 0 || queries.dim != database.dim) {
    throw std::invalid_argument("query dimensions do not match IVF index");
  }
  std::vector<std::vector<std::int64_t>> candidate_ids(
      static_cast<std::size_t>(queries.count));
  std::vector<double> coarse_latency(static_cast<std::size_t>(queries.count));

  for (std::int64_t query_index = 0; query_index < queries.count;
       ++query_index) {
    const auto start = std::chrono::steady_clock::now();
    const float* query = queries.values.data() + query_index * queries.dim;
    const std::vector<int> lists = ranked_centers(query, index);
    auto& candidates = candidate_ids[static_cast<std::size_t>(query_index)];
    int probed = 0;
    while (probed < index.nlist &&
           (probed < config.nprobe ||
            candidates.size() < static_cast<std::size_t>(config.top_k))) {
      const int list = lists[static_cast<std::size_t>(probed++)];
      const auto begin = index.offsets[static_cast<std::size_t>(list)];
      const auto end = index.offsets[static_cast<std::size_t>(list + 1)];
      candidates.insert(candidates.end(), index.ids.begin() + begin,
                        index.ids.begin() + end);
    }
    // CUB radix sort 对相同 score 保持输入顺序；先按全局 vector ID 排序，
    // 使 CUDA 与 CPU 都实现“score 相同则较小 ID 优先”的确定性规则。
    std::sort(candidates.begin(), candidates.end());
    const auto end = std::chrono::steady_clock::now();
    coarse_latency[static_cast<std::size_t>(query_index)] =
        std::chrono::duration<double, std::milli>(end - start).count();
  }

  SearchResults results =
      use_cuda
          ? search_candidates_cuda(database, queries, candidate_ids,
                                   config.top_k, config.batch_size)
          : search_candidates_cpu(database, queries, candidate_ids,
                                  config.top_k, config.batch_size);
  for (std::size_t query = 0; query < results.query_latency_ms.size(); ++query) {
    results.query_latency_ms[query] += coarse_latency[query];
  }
  results.backend = use_cuda ? "cuda-ivf-flat" : "cpu-ivf-flat";
  return results;
}

}  // namespace vector_search
