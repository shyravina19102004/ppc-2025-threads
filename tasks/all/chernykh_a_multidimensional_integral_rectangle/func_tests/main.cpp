#include <gtest/gtest.h>

#include <boost/mpi/communicator.hpp>
#include <cmath>
#include <cstdint>
#include <memory>
#include <numbers>
#include <vector>

#include "all/chernykh_a_multidimensional_integral_rectangle/include/ops_all.hpp"
#include "core/task/include/task.hpp"

namespace {

using namespace chernykh_a_multidimensional_integral_rectangle_all;

void RunValidTask(const Function& func, std::vector<Dimension>& dims, double want) {
  auto world = boost::mpi::communicator();

  double output = 0.0;
  auto task_data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(dims.data()));
    task_data->inputs_count.emplace_back(dims.size());
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(&output));
    task_data->outputs_count.emplace_back(1);
  }
  auto task = AllTask(task_data, func);

  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());
  if (world.rank() == 0) {
    EXPECT_NEAR(want, output, 1e-8);
  }
}

void RunInvalidTask(const Function& func, std::vector<Dimension>& dims) {
  auto world = boost::mpi::communicator();

  double output = 0.0;
  auto task_data = std::make_shared<ppc::core::TaskData>();
  if (world.rank() == 0) {
    task_data->inputs.emplace_back(reinterpret_cast<uint8_t*>(dims.data()));
    task_data->inputs_count.emplace_back(dims.size());
    task_data->outputs.emplace_back(reinterpret_cast<uint8_t*>(&output));
    task_data->outputs_count.emplace_back(1);
    auto task = AllTask(task_data, func);

    ASSERT_FALSE(task.Validation());
  }
}

TEST(chernykh_a_multidimensional_integral_rectangle_all, radical_2d_integration) {
  Function func = [](const Point& point) -> double { return std::sqrt(point[0]) + std::sqrt(point[1]); };
  std::vector<Dimension> dims = {
      Dimension(0.0, 4.0, 10),
      Dimension(0.0, 9.0, 10),
  };
  double want = 127.89168150722712;
  RunValidTask(func, dims, want);
}

TEST(chernykh_a_multidimensional_integral_rectangle_all, absolute_2d_integration) {
  Function func = [](const Point& point) -> double { return std::abs(point[0]) + std::abs(point[1]); };
  std::vector<Dimension> dims = {
      Dimension(-2.0, 2.0, 10),
      Dimension(-3.0, 3.0, 15),
  };
  double want = 60.16000000000002;
  RunValidTask(func, dims, want);
}

TEST(chernykh_a_multidimensional_integral_rectangle_all, trigonometric_2d_integration) {
  Function func = [](const Point& point) -> double { return std::sin(point[0]) * std::cos(point[1]); };
  std::vector<Dimension> dims = {
      Dimension(0.0, std::numbers::pi, 100),
      Dimension(0.0, std::numbers::pi / 2, 100),
  };
  double want = 1.9840877124304817;
  RunValidTask(func, dims, want);
}

TEST(chernykh_a_multidimensional_integral_rectangle_all, polynomial_3d_integration) {
  Function func = [](const Point& point) -> double {
    return (point[0] * point[1]) + (point[1] * point[2]) + (point[0] * point[2]);
  };
  std::vector<Dimension> dims = {
      Dimension(0.0, 100.0, 50),
      Dimension(-100.0, 0.0, 50),
      Dimension(-50.0, 50.0, 50),
  };
  double want = -2497000000.0;
  RunValidTask(func, dims, want);
}

TEST(chernykh_a_multidimensional_integral_rectangle_all, exponential_1d_integration) {
  Function func = [](const Point& point) -> double { return std::exp(point[0]); };
  std::vector<Dimension> dims = {
      Dimension(0.0, 1.0, 5),
  };
  double want = 1.8958338026286925;
  RunValidTask(func, dims, want);
}

TEST(chernykh_a_multidimensional_integral_rectangle_all, trigonometric_3d_integration) {
  Function func = [](const Point& point) -> double {
    return std::sin(point[0]) * std::cos(point[1]) * std::tan(point[2]);
  };
  std::vector<Dimension> dims = {
      Dimension(0.0, std::numbers::pi, 15),
      Dimension(0.0, std::numbers::pi / 2, 10),
      Dimension(0.0, std::numbers::pi / 4, 5),
  };
  double want = 0.782587506841825;
  RunValidTask(func, dims, want);
}

TEST(chernykh_a_multidimensional_integral_rectangle_all, cubic_2d_integration) {
  Function func = [](const Point& point) -> double { return std::pow(point[0], 3) + std::pow(point[1], 3); };
  std::vector<Dimension> dims = {
      Dimension(0.0, 1.0, 5),
      Dimension(0.0, 2.0, 5),
  };
  double want = 6.480000000000001;
  RunValidTask(func, dims, want);
}

TEST(chernykh_a_multidimensional_integral_rectangle_all, quadratic_2d_integration) {
  Function func = [](const Point& point) -> double { return std::pow(point[0], 2) + std::pow(point[1], 2); };
  std::vector<Dimension> dims = {
      Dimension(0.0, 0.000002, 150),
      Dimension(0.0, 0.000003, 150),
  };
  double want = 2.6260577777777697e-23;
  RunValidTask(func, dims, want);
}

TEST(chernykh_a_multidimensional_integral_rectangle_all, exponential_3d_integration) {
  Function func = [](const Point& point) -> double { return std::exp(point[0] + point[1] + point[2]); };
  std::vector<Dimension> dims = {
      Dimension(0.0, 0.005, 50),
      Dimension(0.0, 0.005, 50),
      Dimension(0.0, 0.005, 50),
  };
  double want = 1.2596031046900125e-07;
  RunValidTask(func, dims, want);
}

TEST(chernykh_a_multidimensional_integral_rectangle_all, trigonometric_exponential_3d_integration) {
  Function func = [](const Point& point) -> double {
    return std::sin(point[0]) * std::cos(point[1]) * std::exp(point[2]);
  };
  std::vector<Dimension> dims = {
      Dimension(0.0, std::numbers::pi, 30),
      Dimension(0.0, std::numbers::pi / 2, 20),
      Dimension(0.0, 1.0, 10),
  };
  double want = 3.4644155398722236;
  RunValidTask(func, dims, want);
}

TEST(chernykh_a_multidimensional_integral_rectangle_all, linear_4d_integration) {
  Function func = [](const Point& point) -> double { return point[0] + point[1] + point[2] + point[3]; };
  std::vector<Dimension> dims = {
      Dimension(0.0, 1.0, 10),
      Dimension(0.0, 1.0, 10),
      Dimension(0.0, 1.0, 10),
      Dimension(0.0, 1.0, 10),
  };
  double want = 2.2000000000000006;
  RunValidTask(func, dims, want);
}

TEST(chernykh_a_multidimensional_integral_rectangle_all, quadratic_4d_integration) {
  Function func = [](const Point& point) -> double {
    return std::pow(point[0], 2) + std::pow(point[1], 2) + std::pow(point[2], 2) + std::pow(point[3], 2);
  };
  std::vector<Dimension> dims = {
      Dimension(0.0, 1.0, 10),
      Dimension(0.0, 1.0, 10),
      Dimension(0.0, 1.0, 10),
      Dimension(0.0, 1.0, 10),
  };
  double want = 1.5399999999999965;
  RunValidTask(func, dims, want);
}

TEST(chernykh_a_multidimensional_integral_rectangle_all, mixed_4d_integration) {
  Function func = [](const Point& point) -> double { return (point[0] * point[1]) + (point[2] * point[3]); };
  std::vector<Dimension> dims = {
      Dimension(0.0, 1.0, 10),
      Dimension(0.0, 1.0, 10),
      Dimension(0.0, 1.0, 10),
      Dimension(0.0, 1.0, 10),
  };
  double want = 0.6049999999999999;
  RunValidTask(func, dims, want);
}

TEST(chernykh_a_multidimensional_integral_rectangle_all, one_step_integration) {
  Function func = [](const Point& point) -> double { return std::pow(point[0], 2) + std::pow(point[1], 2); };
  std::vector<Dimension> dims = {
      Dimension(0.0, 1.0, 1),
      Dimension(0.0, 1.0, 1),
  };
  double want = 2.0;
  RunValidTask(func, dims, want);
}

TEST(chernykh_a_multidimensional_integral_rectangle_all, empty_dimensions_fails_validation) {
  Function func = [](const Point& point) -> double { return std::sqrt(point[0]) + std::sqrt(point[1]); };
  std::vector<Dimension> dims = {};
  RunInvalidTask(func, dims);
}

TEST(chernykh_a_multidimensional_integral_rectangle_all, incorrect_bounds_fails_validation) {
  Function func = [](const Point& point) -> double { return std::abs(point[0]) + std::abs(point[1]); };
  std::vector<Dimension> dims = {
      Dimension(2.0, -2.0, 10),
      Dimension(-3.0, 3.0, 15),
  };
  RunInvalidTask(func, dims);
}

TEST(chernykh_a_multidimensional_integral_rectangle_all, incorrect_steps_count_fails_validation) {
  Function func = [](const Point& point) -> double { return std::sin(point[0]) * std::cos(point[1]); };
  std::vector<Dimension> dims = {
      Dimension(0.0, std::numbers::pi, -100),
      Dimension(0.0, std::numbers::pi / 2, 100),
  };
  RunInvalidTask(func, dims);
}

}  // namespace
