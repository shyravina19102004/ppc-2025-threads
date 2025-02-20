#include "seq/shuravina_o_hoare_simple_merger/include/ops_seq.hpp"

#include <algorithm>
#include <iostream>

bool shuravina_o_hoare_simple_merger::HoareSortSimpleMerge::PreProcessingImpl() {
  auto* in_ptr = reinterpret_cast<int*>(task_data->inputs[0]);
  input_ = std::vector<int>(in_ptr, in_ptr + task_data->inputs_count[0]);

  output_.resize(task_data->outputs_count[0]);
  return true;
}

bool shuravina_o_hoare_simple_merger::HoareSortSimpleMerge::ValidationImpl() {
  return task_data->inputs_count[0] == task_data->outputs_count[0];
}

bool shuravina_o_hoare_simple_merger::HoareSortSimpleMerge::RunImpl() {
  QuickSort(input_, 0, input_.size() - 1);
  output_ = input_;
  return true;
}

bool shuravina_o_hoare_simple_merger::HoareSortSimpleMerge::PostProcessingImpl() {
  for (size_t i = 0; i < output_.size(); i++) {
    reinterpret_cast<int*>(task_data->outputs[0])[i] = output_[i];
  }
  return true;
}

void shuravina_o_hoare_simple_merger::HoareSortSimpleMerge::QuickSort(std::vector<int>& arr, int low, int high) {
  if (low < high) {
    int pi = Partition(arr, low, high);
    QuickSort(arr, low, pi - 1);
    QuickSort(arr, pi + 1, high);
  }
}

int shuravina_o_hoare_simple_merger::HoareSortSimpleMerge::Partition(std::vector<int>& arr, int low, int high) {
  int pivot = arr[high];
  int i = (low - 1);

  for (int j = low; j <= high - 1; j++) {
    if (arr[j] < pivot) {
      i++;
      std::swap(arr[i], arr[j]);
    }
  }
  std::swap(arr[i + 1], arr[high]);
  return (i + 1);
}