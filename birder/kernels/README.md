# Kernels

This directory contains Birder's native C++ and CUDA extensions.

## Development Guidelines

- Use `snake_case` for files, functions, variables and parameters.
- Use `at::Tensor` in native APIs. Pass read-only tensors as `const at::Tensor &`, mutable tensors as `at::Tensor &` and scalar values by value.
- Keep headers self-contained and include their direct dependencies. Prefer focused ATen and c10 headers over broad Torch headers outside binding translation units.
- Keep dispatcher registration in `op.cpp`. Define operators with `TORCH_LIBRARY`, register ATen-composed implementations with `c10::DispatchKey::CompositeImplicitAutograd` and native CUDA implementations with `c10::kCUDA`. Intentionally non-differentiable operators may instead use `c10::DispatchKey::CompositeExplicitAutograd` with an explicit autograd-not-implemented fallback. Load extensions with `is_python_module=False`; do not use `PYBIND11_MODULE`.
- Use `TORCH_CHECK` for errors exposed through PyTorch.
- Keep validation on the hot path minimal and constant-time. Validate only conditions whose violation could silently produce incorrect results, corrupt state, or introduce a race condition. Do not duplicate checks already enforced promptly by dispatch, tensor accessors, views, or other ATen operations. Let obviously nonsensical inputs, such as negative lengths or non-positive step sizes and conditions that already fail promptly fail naturally. Do not synchronize or inspect tensor contents for validation.
- CUDA entry points must guard the input device, launch on PyTorch's current CUDA stream and call `C10_CUDA_KERNEL_LAUNCH_CHECK()` after every kernel launch.

## Testing Strategy

Kernel tests exercise the native extension directly, while operator tests exercise the public Python API. Keep the
responsibilities split as follows:

| Concern                                                   | Kernel tests                    | Operator tests                           |
|-----------------------------------------------------------|---------------------------------|------------------------------------------|
| Native forward and backward math                          | Primary coverage                | Representative parity only               |
| Output shape, dtype and device                            | Exhaustive coverage             | Public-facing smoke coverage             |
| Supported input dtypes and layouts                        | Yes                             | Wrapper-specific conversions only        |
| Explicit `TORCH_CHECK` validation                         | Yes                             | Do not duplicate                         |
| Wrapper-only Python validation                            | No                              | Yes                                      |
| CUDA stream and device behavior                           | Yes                             | No                                       |
| Fallback implementation                                   | No                              | Yes                                      |
| Accelerated/fallback path selection                       | No                              | Yes                                      |
| Wrapper batching, squeezing, class offsets and packing    | No                              | Yes                                      |
| Autograd                                                  | Raw gradient math               | Registration and end-to-end gradients    |
| Fake tensors, `torch.library.opcheck` and compilation     | No                              | Yes, where registered                    |
| Autocast policy                                           | TBD                             | TBD                                      |

For each native entry point, organize independent tests in this order: tensor properties, numerical agreement with a small deterministic reference, backward agreement when applicable, meaningful edge cases and deliberate native validation.
Use explicit tolerances per dtype and compare complete outputs rather than qualitative properties.
Keep regular unit-test inputs small, production-size and performance coverage belong in benchmarks or dedicated tests.

## Adding a Kernel

1. Create `birder/kernels/<kernel_name>/` and put dispatcher registration in `op.cpp`.
2. Add a `KernelSpec` with the complete source list, include paths and compiler flags, plus a lazy loader function.
3. Register the operator schema and implementation through `TORCH_LIBRARY` using the appropriate dispatch key.
4. Add the public wrapper under `birder/ops/`, including a fallback when practical.
5. Add small numerical correctness tests for outputs and gradients where applicable.
6. Run `invoke clang-format` and the relevant kernel and operator tests.
