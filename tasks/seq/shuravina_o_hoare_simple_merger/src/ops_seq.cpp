#include "seq/shuravina_o_hoare_simple_merger/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

bool shuravina_o_hoare_simple_merger::HoareSortSimpleMerge::PreProcessingImpl() {
  unsigned int input_size = task_data->inputs_count[0];
  auto *in_ptr = reinterpret_cast<int *>(task_data->inputs[0]);
  input_ = std::vector<int>(in_ptr, in_ptr + input_size);

  unsigned int output_size = task_data->outputs_count[0];
  output_ = std::vector<int>(output_size, 0);

  return true;
}

bool shuravina_o_hoare_simple_merger::HoareSortSimpleMerge::ValidationImpl() {
  return task_data->inputs_count[0] == task_data->outputs_count[0];
}

void shuravina_o_hoare_simple_merger::HoareSortSimpleMerge::QuickSort(std::vector<int> &arr, int low, int high) {
  if (low < high) {
    int pivot = arr[high];
    int i = low - 1;
    for (int j = low; j < high; j++) {
      if (arr[j] <= pivot) {
        i++;
        std::swap(arr[i], arr[j]);
      }
    }
    std::swap(arr[i + 1], arr[high]);
    int pi = i + 1;

    QuickSort(arr, low, pi - 1);
    QuickSort(arr, pi + 1, high);
  }
}

void shuravina_o_hoare_simple_merger::HoareSortSimpleMerge::Merge(std::vector<int> &arr, int low, int mid, int high) {
  int n1 = mid - low + 1;
  int n2 = high - mid;

  std::vector<int> left(n1), right(n2);

  for (int i = 0; i < n1; i++) left[i] = arr[low + i];
  for (int i = 0; i < n2; i++) right[i] = arr[mid + 1 + i];

  int i = 0, j = 0, k = low;
  while (i < n1 && j < n2) {
    if (left[i] <= right[j]) {
      arr[k] = left[i];
      i++;
    } else {
      arr[k] = right[j];
      j++;
    }
    k++;
  }

  while (i < n1) {
    arr[k] = left[i];
    i++;
    k++;
  }

  while (j < n2) {
    arr[k] = right[j];
    j++;
    k++;
  }
}

bool shuravina_o_hoare_simple_merger::HoareSortSimpleMerge::RunImpl() {
  QuickSort(input_, 0, input_.size() - 1);
  output_ = input_;
  return true;
}

bool shuravina_o_hoare_simple_merger::HoareSortSimpleMerge::PostProcessingImpl() {
  for (size_t i = 0; i < output_.size(); i++) {
    reinterpret_cast<int *>(task_data->outputs[0])[i] = output_[i];
  }
  return true;
}