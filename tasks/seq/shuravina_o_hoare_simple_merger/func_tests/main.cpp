#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "core/task/include/task.hpp"
#include "core/util/include/util.hpp"
#include "seq/shuravina_o_hoare_simple_merger/include/ops_seq.hpp"

TEST(shuravina_o_hoare_simple_merger_seq, test_sort_50) {
  constexpr size_t kCount = 50;

  std::vector<int> in(kCount, 0);
  std::vector<int> out(kCount, 0);

  for (size_t i = 0; i < kCount; i++) {
    in[i] = static_cast<int>(kCount - i);
  }

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  shuravina_o_hoare_simple_merger::HoareSortSimpleMerge test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();

  std::vector<int> expected(kCount);
  for (size_t i = 0; i < kCount; i++) {
    expected[i] = static_cast<int>(i + 1);
  }
  EXPECT_EQ(out, expected);
}

TEST(shuravina_o_hoare_simple_merger_seq, test_sort_100_from_file) {
  std::string line;
  std::ifstream test_file(ppc::util::GetAbsolutePath("seq/shuravina_o_hoare_simple_merger/data/test.txt"));
  if (test_file.is_open()) {
    getline(test_file, line);
  }
  test_file.close();

  const size_t count = std::stoi(line);

  std::vector<int> in(count, 0);
  std::vector<int> out(count, 0);

  for (size_t i = 0; i < count; i++) {
    in[i] = static_cast<int>(count - i);
  }

  auto task_data_seq = std::make_shared<ppc::core::TaskData>();
  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(in.data()));
  task_data_seq->inputs_count.emplace_back(in.size());
  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  shuravina_o_hoare_simple_merger::HoareSortSimpleMerge test_task_sequential(task_data_seq);
  ASSERT_EQ(test_task_sequential.Validation(), true);
  test_task_sequential.PreProcessing();
  test_task_sequential.Run();
  test_task_sequential.PostProcessing();

  std::vector<int> expected(count);
  for (size_t i = 0; i < count; i++) {
    expected[i] = static_cast<int>(i + 1);
  }
  EXPECT_EQ(out, expected);
}