#pragma once
#include <torch/types.h>

namespace pet {

at::Tensor ROIAlign_forward_cpu(
    const at::Tensor& input,
    const at::Tensor& rois,
    const float spatial_scale,
    const int pooled_height,
    const int pooled_width,
    const int sampling_ratio,
    bool aligned,
    const int interpolation_method);

at::Tensor ROIAlign_backward_cpu(
    const at::Tensor& grad,
    const at::Tensor& rois,
    const float spatial_scale,
    const int pooled_height,
    const int pooled_width,
    const int batch_size,
    const int channels,
    const int height,
    const int width,
    const int sampling_ratio,
    bool aligned,
    const int interpolation_method);

#ifdef WITH_CUDA
at::Tensor ROIAlign_forward_cuda(
    const at::Tensor& input,
    const at::Tensor& rois,
    const float spatial_scale,
    const int pooled_height,
    const int pooled_width,
    const int sampling_ratio,
    const bool aligned,
    const int interpolation_method);

at::Tensor ROIAlign_backward_cuda(
    const at::Tensor& grad,
    const at::Tensor& rois,
    const float spatial_scale,
    const int pooled_height,
    const int pooled_width,
    const int batch_size,
    const int channels,
    const int height,
    const int width,
    const int sampling_ratio,
    const bool aligned,
    const int interpolation_method);
#endif

// Interface for Python
inline at::Tensor ROIAlign_forward(
    const at::Tensor& input,
    const at::Tensor& rois,
    const float spatial_scale,
    const int pooled_height,
    const int pooled_width,
    const int sampling_ratio,
    const bool aligned,
    const int interpolation_method) {
  AT_ASSERTM(
    interpolation_method == 0 || interpolation_method == 1,
    "interpolation must be bilinear or nearest");
  if (input.type().is_cuda()) {
#ifdef WITH_CUDA
    return ROIAlign_forward_cuda(
        input,
        rois,
        spatial_scale,
        pooled_height,
        pooled_width,
        sampling_ratio,
        aligned,
        interpolation_method);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  }
  AT_ASSERTM(
    interpolation_method == 0,
    "interpolation must be bilinear with CPU");
  return ROIAlign_forward_cpu(
      input,
      rois,
      spatial_scale,
      pooled_height,
      pooled_width,
      sampling_ratio,
      aligned,
      interpolation_method);
}

inline at::Tensor ROIAlign_backward(
    const at::Tensor& grad,
    const at::Tensor& rois,
    const float spatial_scale,
    const int pooled_height,
    const int pooled_width,
    const int batch_size,
    const int channels,
    const int height,
    const int width,
    const int sampling_ratio,
    const bool aligned,
    const int interpolation_method) {
  AT_ASSERTM(
    interpolation_method == 0 || interpolation_method == 1,
    "interpolation must be bilinear or nearest");
  if (grad.type().is_cuda()) {
#ifdef WITH_CUDA
    return ROIAlign_backward_cuda(
        grad,
        rois,
        spatial_scale,
        pooled_height,
        pooled_width,
        batch_size,
        channels,
        height,
        width,
        sampling_ratio,
        aligned,
        interpolation_method);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  }
  return ROIAlign_backward_cpu(
      grad,
      rois,
      spatial_scale,
      pooled_height,
      pooled_width,
      batch_size,
      channels,
      height,
      width,
      sampling_ratio,
      aligned,
      interpolation_method);
}

} // namespace pet
