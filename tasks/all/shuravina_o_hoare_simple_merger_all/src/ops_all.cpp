#include "all/shuravina_o_hoare_simple_merger_all/include/ops_all.hpp"

#include <mpi.h>
#include <omp.h>

#include <algorithm>
#include <cstddef>
#include <vector>

namespace shuravina_o_hoare_simple_merger {

void TestTaskALL::QuickSort(std::vector<int>& arr, int low, int high) {
  if (low < high) {
    bool is_sorted = true;
    for (int i = low + 1; i <= high; ++i) {
      if (arr[i - 1] > arr[i]) {
        is_sorted = false;
        break;
      }
    }
    if (is_sorted) {
      return;
    }

    int pivot = arr[high];
    int i = low - 1;

    for (int j = low; j < high; ++j) {
      if (arr[j] <= pivot) {
        ++i;
        std::swap(arr[i], arr[j]);
      }
    }
    std::swap(arr[i + 1], arr[high]);

    int pi = i + 1;

#pragma omp parallel sections
    {
#pragma omp section
      QuickSort(arr, low, pi - 1);
#pragma omp section
      QuickSort(arr, pi + 1, high);
    }
  }
}

void TestTaskALL::Merge(std::vector<int>& arr, int low, int mid, int high) {
  std::vector<int> temp(high - low + 1);
  int i = low;
  int j = mid + 1;
  int k = 0;

  while (i <= mid && j <= high) {
    if (arr[i] <= arr[j]) {
      temp[k++] = arr[i++];
    } else {
      temp[k++] = arr[j++];
    }
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

bool TestTaskALL::PreProcessingImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0) {
    if (task_data->inputs.empty() || task_data->inputs[0] == nullptr) {
      return false;
    }

    auto* in_ptr = reinterpret_cast<int*>(task_data->inputs[0]);
    auto input_size = static_cast<size_t>(task_data->inputs_count[0]);
    input_ = std::vector<int>(in_ptr, in_ptr + input_size);

    if (task_data->outputs.empty() || task_data->outputs[0] == nullptr) {
      return false;
    }

    auto output_size = static_cast<size_t>(task_data->outputs_count[0]);
    output_ = std::vector<int>(output_size, 0);
  }

  return true;
}

bool TestTaskALL::ValidationImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0) {
    if (task_data->inputs.empty() || task_data->outputs.empty()) {
      return false;
    }

    if (task_data->inputs[0] == nullptr || task_data->outputs[0] == nullptr) {
      return false;
    }

    if (task_data->inputs_count.empty() || task_data->outputs_count.empty()) {
      return false;
    }

    if (task_data->inputs_count[0] != task_data->outputs_count[0]) {
      return false;
    }
  }

  return true;
}

bool TestTaskALL::RunImpl() {
  int rank = 0;
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (input_.empty()) {
    return true;
  }

  if (rank == 0) {
    auto arr_size = input_.size();
    QuickSort(input_, 0, static_cast<int>(arr_size) - 1);
    Merge(input_, 0, static_cast<int>(arr_size / 2) - 1, static_cast<int>(arr_size) - 1);
    output_ = input_;
  }

  return true;
}

bool TestTaskALL::PostProcessingImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0) {
    if (output_.empty() || task_data->outputs[0] == nullptr) {
      return false;
    }

    auto* out_ptr = reinterpret_cast<int*>(task_data->outputs[0]);
    for (size_t i = 0; i < output_.size(); i++) {
      out_ptr[i] = output_[i];
    }
  }

  return true;
}

}  // namespace shuravina_o_hoare_simple_merger