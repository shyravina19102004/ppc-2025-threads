#pragma once

#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace shuravina_o_hoare_simple_merger {

class HoareSortSimpleMerge : public ppc::core::Task {
 public:
  explicit HoareSortSimpleMerge(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  void QuickSort(std::vector<int>& arr, int low, int high);
  int Partition(std::vector<int>& arr, int low, int high);

  std::vector<int> input_, output_;
};

}  // namespace shuravina_o_hoare_simple_merger