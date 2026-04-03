/*
  Implementation is based on the algorithm presented in pages 1685-1686 of:

  DF Crouse. On implementing 2D rectangular assignment algorithms.
    IEEE Transactions on Aerospace and Electronic Systems
    52(4):1679-1696, August 2016
    doi: 10.1109/TAES.2016.140952

* Modified by:
* Ofer Hasson — 2026-03-27
*/

#include <cuda.h>
#include <cuda_runtime.h>

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include <cstdint>
#include <limits>
#include <vector>

typedef unsigned char uint8_t;

int SMPCores(int device_index)
{
  cudaDeviceProp devProp;
  cudaGetDeviceProperties(&devProp, device_index);
  switch (devProp.major)
  {
  case 2:
    if (devProp.minor == 1)
    {
      return 48;
    }
    return 32;
  case 3:
    return 192;
  case 5:
    return 128;
  case 6:
    if ((devProp.minor == 1) || (devProp.minor == 2))
    {
      return 128;
    }
    else if (devProp.minor == 0)
    {
      return 64;
    }
    break;
  case 7:
    if ((devProp.minor == 0) || (devProp.minor == 5))
    {
      return 64;
    }
    break;
  case 8:
    if (devProp.minor == 0)
    {
      return 64;
    }
    else if (devProp.minor == 6)
    {
      return 128;
    }
    else if (devProp.minor == 9)
    {
      return 128;
    }
    break;
  case 9:
    if (devProp.minor == 0)
    {
      return 128;
    }
    break;
  }

  return 128;
}

template <typename scalar_t>
__device__ __forceinline__ void array_fill(scalar_t *start, scalar_t *stop, scalar_t value)
{
  for (; start < stop; ++start)
  {
    *start = value;
  }
}

template <typename scalar_t>
__device__ __forceinline__ int augmenting_path_cuda(int nr, int nc, int i,
                                                    const scalar_t *cost,
                                                    int64_t cost_row_stride, int64_t cost_col_stride,
                                                    scalar_t *u, scalar_t *v,
                                                    int *path, int *row4col,
                                                    scalar_t *shortestPathCosts,
                                                    uint8_t *SR, uint8_t *SC,
                                                    int *remaining,
                                                    scalar_t *p_minVal,
                                                    scalar_t infinity)
{
  scalar_t minVal = 0;
  int num_remaining = nc;
  for (int it = 0; it < nc; ++it)
  {
    SC[it] = 0;
    remaining[it] = nc - it - 1;
    shortestPathCosts[it] = infinity;
  }

  array_fill(SR, SR + nr, static_cast<uint8_t>(0));

  int sink = -1;
  while (sink == -1)
  {
    int index = -1;
    scalar_t lowest = infinity;
    SR[i] = 1;

    const scalar_t *cost_row = cost + static_cast<int64_t>(i) * cost_row_stride;
    scalar_t base_r = minVal - u[i];
    for (int it = 0; it < num_remaining; ++it)
    {
      int j = remaining[it];
      scalar_t r = base_r + cost_row[static_cast<int64_t>(j) * cost_col_stride] - v[j];
      if (r < shortestPathCosts[j])
      {
        path[j] = i;
        shortestPathCosts[j] = r;
      }
      if (shortestPathCosts[j] < lowest ||
          (shortestPathCosts[j] == lowest && row4col[j] == -1))
      {
        lowest = shortestPathCosts[j];
        index = it;
      }
    }

    minVal = lowest;
    if (minVal == infinity)
    {
      return -1;
    }

    int j = remaining[index];
    if (row4col[j] == -1)
    {
      sink = j;
    }
    else
    {
      i = row4col[j];
    }

    SC[j] = 1;
    remaining[index] = remaining[--num_remaining];
  }

  *p_minVal = minVal;
  return sink;
}

template <typename scalar_t>
__device__ __forceinline__ void solve_cuda_kernel(int nr, int nc,
                                                  const scalar_t *cost,
                                                  int64_t cost_row_stride, int64_t cost_col_stride,
                                                  scalar_t *u, scalar_t *v,
                                                  scalar_t *shortestPathCosts,
                                                  int *path, int *col4row, int *row4col,
                                                  uint8_t *SR, uint8_t *SC,
                                                  int *remaining,
                                                  scalar_t infinity)
{
  scalar_t minVal;
  for (int cur_row = 0; cur_row < nr; ++cur_row)
  {
    int sink = augmenting_path_cuda(nr, nc, cur_row, cost,
                                    cost_row_stride, cost_col_stride,
                                    u, v,
                                    path, row4col,
                                    shortestPathCosts,
                                    SR, SC,
                                    remaining,
                                    &minVal, infinity);

    CUDA_KERNEL_ASSERT(sink >= 0 && "Infeasible matrix");

    u[cur_row] += minVal;
    for (int i = 0; i < nr; ++i)
    {
      if (SR[i] && i != cur_row)
      {
        u[i] += minVal - shortestPathCosts[col4row[i]];
      }
    }

    for (int j = 0; j < nc; ++j)
    {
      if (SC[j])
      {
        v[j] -= minVal - shortestPathCosts[j];
      }
    }

    int i = -1;
    int j = sink;
    int swap;
    while (i != cur_row)
    {
      i = path[j];
      row4col[j] = i;
      swap = j;
      j = col4row[i];
      col4row[i] = swap;
    }
  }
}

template <typename scalar_t>
__global__ void solve_cuda_kernel_batch(int bs, int nr, int nc,
                                        const scalar_t *cost,
                                        int64_t cost_batch_stride,
                                        int64_t cost_row_stride, int64_t cost_col_stride,
                                        scalar_t *u, scalar_t *v,
                                        scalar_t *shortestPathCosts,
                                        int *path, int *col4row, int *row4col,
                                        uint8_t *SR, uint8_t *SC,
                                        int *remaining,
                                        scalar_t infinity)
{
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i >= bs)
  {
    return;
  }

  solve_cuda_kernel(nr, nc,
                    cost + static_cast<int64_t>(i) * cost_batch_stride,
                    cost_row_stride,
                    cost_col_stride,
                    u + i * nr,
                    v + i * nc,
                    shortestPathCosts + i * nc,
                    path + i * nc,
                    col4row + i * nr,
                    row4col + i * nc,
                    SR + i * nr,
                    SC + i * nc,
                    remaining + i * nc,
                    infinity);
}

template <typename scalar_t>
void solve_cuda_batch(c10::ScalarType scalar_type,
                      int device_index,
                      int bs, int nr, int nc,
                      const scalar_t *cost,
                      int64_t cost_batch_stride,
                      int64_t cost_row_stride,
                      int64_t cost_col_stride,
                      int *col4row, int *row4col)
{
  TORCH_CHECK(std::numeric_limits<scalar_t>::has_infinity, "Data type doesn't have infinity.");
  auto infinity = std::numeric_limits<scalar_t>::infinity();

  auto int_opt = torch::TensorOptions().dtype(torch::kInt).device(torch::kCUDA, device_index);
  auto scalar_t_opt = torch::TensorOptions().dtype(scalar_type).device(torch::kCUDA, device_index);
  auto uint8_opt = torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA, device_index);

  torch::Tensor u = torch::zeros({bs * nr}, scalar_t_opt);
  torch::Tensor v = torch::zeros({bs * nc}, scalar_t_opt);
  torch::Tensor shortest_path_costs = torch::empty({bs * nc}, scalar_t_opt);
  torch::Tensor path = torch::empty({bs * nc}, int_opt);
  torch::Tensor SR = torch::empty({bs * nr}, uint8_opt);
  torch::Tensor SC = torch::empty({bs * nc}, uint8_opt);
  torch::Tensor remaining = torch::empty({bs * nc}, int_opt);

  static const int block_size = SMPCores(device_index);
  int grid_size = (bs + block_size - 1) / block_size;
  at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream(device_index);
  solve_cuda_kernel_batch<<<grid_size, block_size, 0, stream.stream()>>>(
      bs, nr, nc,
      cost,
      cost_batch_stride,
      cost_row_stride,
      cost_col_stride,
      u.data_ptr<scalar_t>(),
      v.data_ptr<scalar_t>(),
      shortest_path_costs.data_ptr<scalar_t>(),
      path.data_ptr<int>(),
      col4row,
      row4col,
      SR.data_ptr<uint8_t>(),
      SC.data_ptr<uint8_t>(),
      remaining.data_ptr<int>(),
      infinity);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    TORCH_CHECK(false, cudaGetErrorString(err));
  }
}

std::vector<torch::Tensor> batch_linear_assignment_cuda_impl(const torch::Tensor &cost, bool transpose_cost)
{
  auto sizes = cost.sizes();
  auto strides = cost.strides();

  int batch_size = static_cast<int>(sizes[0]);
  int nr = static_cast<int>(transpose_cost ? sizes[2] : sizes[1]);
  int nc = static_cast<int>(transpose_cost ? sizes[1] : sizes[2]);
  int64_t cost_batch_stride = strides[0];
  int64_t cost_row_stride = transpose_cost ? strides[2] : strides[1];
  int64_t cost_col_stride = transpose_cost ? strides[1] : strides[2];

  auto device = cost.device();
  auto options = torch::TensorOptions().dtype(torch::kInt).device(device.type(), device.index());
  torch::Tensor col4row = torch::full({batch_size, nr}, -1, options);
  torch::Tensor row4col = torch::full({batch_size, nc}, -1, options);

  if (batch_size == 0 || nr == 0)
  {
    return {col4row, row4col};
  }

  AT_DISPATCH_FLOATING_TYPES(cost.scalar_type(), "solve_cuda_batch", [&]
                             { solve_cuda_batch<scalar_t>(
                                   cost.scalar_type(),
                                   device.index(),
                                   batch_size, nr, nc,
                                   cost.data_ptr<scalar_t>(),
                                   cost_batch_stride,
                                   cost_row_stride,
                                   cost_col_stride,
                                   col4row.data_ptr<int>(),
                                   row4col.data_ptr<int>()); });

  return {col4row, row4col};
}

std::vector<torch::Tensor> batch_linear_assignment_cuda(const torch::Tensor &cost)
{
  auto sizes = cost.sizes();

  TORCH_CHECK(sizes.size() == 3, "Cost matrix must have shape (B, W, T).");
  if (sizes[1] <= sizes[2])
  {
    return batch_linear_assignment_cuda_impl(cost, false);
  }

  auto assignment = batch_linear_assignment_cuda_impl(cost, true);
  return {assignment[1], assignment[0]};
}
