#include "all/shuravina_o_hoare_simple_merger_all/include/ops_all.hpp"

#include <mpi.h>
#include <omp.h>

#include <algorithm>
#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

namespace shuravina_o_hoare_simple_merger {

TestTaskALL::TestTaskALL(std::shared_ptr<ppc::core::TaskData> task_data) : Task(std::move(task_data)) {}

void TestTaskALL::QuickSort(std::vector<int>& arr, int low, int high) {
  if (low >= high) {
    return;
  }

  int pivot = arr[high];
  int i = low - 1;

  for (int j = low; j < high; ++j) {
    if (arr[j] <= pivot) {
      std::swap(arr[++i], arr[j]);
    }
  }
  std::swap(arr[i + 1], arr[high]);

  int pi = i + 1;

#pragma omp parallel sections
  {
#pragma omp section
    {
      if (pi - 1 > low) {
        QuickSort(arr, low, pi - 1);
      }
    }
#pragma omp section
    {
      if (high > pi + 1) {
        QuickSort(arr, pi + 1, high);
      }
    }
  }
}

void TestTaskALL::Merge(std::vector<int>& arr, int low, int mid, int high) {
  std::vector<int> temp(high - low + 1);
  int i = low;
  int j = mid + 1;
  int k = 0;

  while (i <= mid && j <= high) {
    temp[k++] = arr[i] <= arr[j] ? arr[i++] : arr[j++];
  }

  while (i <= mid) {
    temp[k++] = arr[i++];
  }
  while (j <= high) {
    temp[k++] = arr[j++];
  }

  for (i = low, k = 0; i <= high; ++i, ++k) {
    arr[i] = temp[k];
  }
}

void TestTaskALL::ParallelQuickSort(std::vector<int>& arr) {
  if (!arr.empty()) {
    QuickSort(arr, 0, static_cast<int>(arr.size()) - 1);
  }
}

void TestTaskALL::DistributeData() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (rank == 0) {
    size_t chunk_size = input_.size() / size;
    for (int i = 1; i < size; ++i) {
      size_t start = i * chunk_size;
      size_t end = (i == size - 1) ? input_.size() : (i + 1) * chunk_size;
      size_t count = end - start;
      MPI_Send(&count, 1, MPI_UNSIGNED_LONG, i, 0, MPI_COMM_WORLD);
      MPI_Send(input_.data() + start, static_cast<int>(count), MPI_INT, i, 0, MPI_COMM_WORLD);
    }
    local_data_.assign(input_.begin(), input_.begin() + static_cast<std::ptrdiff_t>(chunk_size));
  } else {
    size_t count = 0;
    MPI_Recv(&count, 1, MPI_UNSIGNED_LONG, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    local_data_.resize(count);
    MPI_Recv(local_data_.data(), static_cast<int>(count), MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }
}

void TestTaskALL::GatherAndMergeResults() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (rank == 0) {
    output_ = local_data_;
    std::vector<int> temp;
    for (int i = 1; i < size; ++i) {
      size_t count = 0;
      MPI_Recv(&count, 1, MPI_UNSIGNED_LONG, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      temp.resize(count);
      MPI_Recv(temp.data(), static_cast<int>(count), MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

      std::vector<int> merged(output_.size() + temp.size());
      std::ranges::merge(output_, temp, merged.begin());
      output_ = std::move(merged);
    }
  } else {
    size_t count = local_data_.size();
    MPI_Send(&count, 1, MPI_UNSIGNED_LONG, 0, 0, MPI_COMM_WORLD);
    MPI_Send(local_data_.data(), static_cast<int>(count), MPI_INT, 0, 0, MPI_COMM_WORLD);
  }
}

bool TestTaskALL::PreProcessingImpl() {
  try {
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) {
      if (!task_data->inputs.empty() && task_data->inputs[0] != nullptr) {
        auto* in_ptr = reinterpret_cast<int*>(task_data->inputs[0]);
        input_.assign(in_ptr, in_ptr + task_data->inputs_count[0]);
        return true;
      }
      return false;
    }
    return true;
  } catch (...) {
    MPI_Abort(MPI_COMM_WORLD, 1);
    return false;
  }
}

bool TestTaskALL::ValidationImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0) {
    return task_data->inputs.size() == 1 && task_data->outputs.size() == 1 && task_data->inputs[0] != nullptr &&
           task_data->outputs[0] != nullptr && !task_data->inputs_count.empty() && !task_data->outputs_count.empty() &&
           task_data->inputs_count[0] == task_data->outputs_count[0];
  }
  return true;
}

bool TestTaskALL::RunImpl() {
  try {
    DistributeData();
    ParallelQuickSort(local_data_);
    GatherAndMergeResults();
    MPI_Barrier(MPI_COMM_WORLD);
    return true;
  } catch (...) {
    MPI_Abort(MPI_COMM_WORLD, 1);
    return false;
  }
}

bool TestTaskALL::PostProcessingImpl() {
  try {
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0 && !output_.empty() && task_data->outputs[0] != nullptr) {
      auto* out_ptr = reinterpret_cast<int*>(task_data->outputs[0]);
      std::ranges::copy(output_, out_ptr);
      return true;
    }
    return rank != 0;
  } catch (...) {
    MPI_Abort(MPI_COMM_WORLD, 1);
    return false;
  }
}

}  // namespace shuravina_o_hoare_simple_merger