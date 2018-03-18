// --------------------------------------------------------
// Multitask Network Cascade
// Written by Haozhi Qi
// Copyright (c) 2016, Haozhi Qi
// Licensed under The MIT License [see LICENSE for details]
// --------------------------------------------------------
#include <cfloat>
#include "caffe/fast_rcnn_layers.hpp"
#include <thrust/reduce.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
using std::max;
using std::min;

namespace caffe {

template <typename Dtype>
__device__ void bilinear_interpolate(const Dtype* bottom_data, const int height, const int width, Dtype h, Dtype w, Dtype & maxval, Dtype & maxidx_h, Dtype & maxidx_w) {
  
  // deal with cases that inverse elements are out of feature map boundary
  if (h < -0.5 || h > height - 0.5 || w < -0.5 || w > width - 0.5) {
    //empty
    return;
  }
  
  if (h <= 0) h = 0;
  if (w <= 0) w = 0;
  
  int h_low = (int) h;
  int w_low = (int) w;
  int h_high;
  int w_high;
  
  if (h_low >= height - 1) {
    h_high = h_low = height - 1;
    h = (Dtype) h_low;
  } else {
    h_high = h_low + 1;
  }
  
  if (w_low >= width - 1) {
    w_high = w_low = width - 1;
    w = (Dtype) w_low;
  } else {
    w_high = w_low + 1;
  }
  
  Dtype lh = h - h_low;
  Dtype lw = w - w_low;
  Dtype hh = 1 - lh, hw = 1 - lw;
  // do bilinear interpolation
  Dtype v1 = bottom_data[h_low * width + w_low];
  Dtype v2 = bottom_data[h_low * width + w_high];
  Dtype v3 = bottom_data[h_high * width + w_low];
  Dtype v4 = bottom_data[h_high * width + w_high];
  Dtype w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;
  
  Dtype val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
  if (val > maxval) {
    maxval = val;
    maxidx_h = h;
    maxidx_w = w;
  }
}
  
template <typename Dtype>
__global__ void ROIAligningForward(const int nthreads, const Dtype* bottom_data,
             const Dtype spatial_scale, const int channels, const int height, const int width,
             const int pooled_height, const int pooled_width, const Dtype* bottom_rois,
             Dtype* top_data, Dtype* argmax_data_h, Dtype* argmax_data_w, int* argmax_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // (n, c, ph, pw) is an element in the pooled output
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;
    
    bottom_rois += n * 5;
    int roi_level = bottom_rois[0];
    Dtype dtype_roi_start_w = bottom_rois[1] * spatial_scale;
    Dtype dtype_roi_start_h = bottom_rois[2] * spatial_scale;
    Dtype dtype_roi_end_w = bottom_rois[3] * spatial_scale;
    Dtype dtype_roi_end_h = bottom_rois[4] * spatial_scale;
    int int_roi_start_w = round(bottom_rois[1] * spatial_scale);
    int int_roi_start_h = round(bottom_rois[2] * spatial_scale);
    int int_roi_end_w = round(bottom_rois[3] * spatial_scale);
    int int_roi_end_h = round(bottom_rois[4] * spatial_scale);

    // Force malformed ROIs to be 1x1
    Dtype dtype_roi_width = max(dtype_roi_end_w - dtype_roi_start_w, (Dtype)0.);
    Dtype dtype_roi_height = max(dtype_roi_end_h - dtype_roi_start_h, (Dtype)0.);
    int int_roi_width = max(int_roi_end_w - int_roi_start_w + 1, 1);
    int int_roi_height = max(int_roi_end_h - int_roi_start_h + 1, 1);

    Dtype bin_size_h = static_cast<Dtype>(dtype_roi_height) / static_cast<Dtype>(pooled_height);
    Dtype bin_size_w = static_cast<Dtype>(dtype_roi_width) / static_cast<Dtype>(pooled_width);

    //calculate the start/end point for the roi after spatial_scale
    Dtype dtype_hstart = static_cast<Dtype>(ph) * bin_size_h;
    Dtype dtype_wstart = static_cast<Dtype>(pw) * bin_size_w;
    Dtype dtype_hend = static_cast<Dtype>(ph + 1) * bin_size_h;
    Dtype dtype_wend = static_cast<Dtype>(pw + 1) * bin_size_w;
    int int_hstart = static_cast<int>(floor(static_cast<Dtype>(ph) * bin_size_h));
    int int_wstart = static_cast<int>(floor(static_cast<Dtype>(pw) * bin_size_w));
    int int_hend = static_cast<int>(ceil(static_cast<Dtype>(ph + 1) * bin_size_h));
    int int_wend = static_cast<int>(ceil(static_cast<Dtype>(pw + 1) * bin_size_w));

    // Add roi offsets and clip to input boundaries
    dtype_hstart = min(max(dtype_hstart + dtype_roi_start_h, Dtype(0)), Dtype(height));
    dtype_hend = min(max(dtype_hend + dtype_roi_start_h, Dtype(0)), Dtype(height));
    dtype_wstart = min(max(dtype_wstart + dtype_roi_start_w, Dtype(0)), Dtype(width));
    dtype_wend = min(max(dtype_wend + dtype_roi_start_w, Dtype(0)), Dtype(width));
    int_hstart = min(max(int_hstart + int_roi_start_h, 0), height);
    int_hend = min(max(int_hend + int_roi_start_h, 0), height);
    int_wstart = min(max(int_wstart + int_roi_start_w, 0), width);
    int_wend = min(max(int_wend + int_roi_start_w, 0), width);


    bool is_empty = (int_hend <= int_hstart) || (int_wend <= int_wstart);

    // Define an empty pooling region to be zero
    Dtype maxval = is_empty ? 0 : -FLT_MAX;
    // Define an empty top_data max_value
    Dtype top_maxval = 0;
    // If nothing is pooled, argmax = -1 causes nothing to be backpropgated
    // Define an empty index for store the height and width value for every interpolatation point
    Dtype maxidx_h = -1;
    Dtype maxidx_w = -1;
    Dtype top_maxidx_h;
    Dtype top_maxidx_w;
    // Define an empty index for store the max value index inside the bin_size_w*bin_size_h
    Dtype top_maxidx;
    //Dtype maxidx = -1;

    bottom_data += (roi_level * channels + c) * height * width;
   
    //int roi_w=floor(static_cast<Dtype>(bin_size_h));
    //int roi_h=floor(static_cast<Dtype>(bin_size_w));
    //Dtype roi_bins_w = static_cast<Dtype>(bin_size_w) / static_cast<int>(roi_w);
    //Dtype roi_bins_h = static_cast<Dtype>(bin_size_h) / static_cast<int>(roi_h);

    for (int h = int_hstart; h < int_hend; ++h) {
      for (int w = int_wstart; w < int_wend; ++w) {
      //do bilinear interpolatation inside bin_size_h*bin_size_w
        Dtype ih = dtype_hstart + h - int_hstart;
        Dtype iw = dtype_wstart + w - int_wstart;
        //maxidx = ih * width + iw;
        int bottom_index = h * width +w;
        bilinear_interpolate(bottom_data, height, width, ih, iw, maxval, maxidx_h, maxidx_w);
        if(maxval>top_maxval){
            top_maxidx_h = maxidx_h;
            top_maxidx_w = maxidx_w;
            top_maxval = maxval;
            top_maxidx = bottom_index;
        }
      }
    }
    top_data[index] = top_maxval;
    argmax_data[index] = top_maxidx;
    argmax_data_h[index] = top_maxidx_h;
    argmax_data_w[index] = top_maxidx_w;
  }
}


template <typename Dtype>
void ROIAligningLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
             const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* bottom_rois = bottom[1]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  Dtype* argmax_data_h = max_idx_h_.mutable_gpu_data();
  Dtype* argmax_data_w = max_idx_w_.mutable_gpu_data();
  int* argmax_data = max_idx_.mutable_gpu_data();
  int count = top[0]->count();

  ROIAligningForward<Dtype> << <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS >> >
    (count, bottom_data, spatial_scale_, channels_, height_, width_, pooled_height_,
     pooled_width_, bottom_rois, top_data, argmax_data_h, argmax_data_w, argmax_data);
  CUDA_POST_KERNEL_CHECK;
  //LOG(INFO)<<"Forward done for roi_align";
}


template <typename Dtype>
__device__ Dtype get_feature_gradient(Dtype argmax_h, Dtype argmax_w, const int h, const int w, const int height, const int width)
{
  if (argmax_h < -0.5 || argmax_h >(height - 0.5) || argmax_w < -0.5 || argmax_w >(width - 0.5))
    {
      //empty
      return 0;
    }
  
  if (argmax_h < 0) argmax_h = 0;
  if (argmax_w < 0) argmax_w = 0;
  
  int argmax_h_low = (int)argmax_h;
  int argmax_w_low = (int)argmax_w;
  int argmax_h_high;
  int argmax_w_high;
  if (argmax_h_low >= height - 1) {
    argmax_h_high = argmax_h_low = height - 1;
    argmax_h = (Dtype)argmax_h_low;
  }
  else
    argmax_h_high = argmax_h_low + 1;
  
  if (argmax_w_low >= width - 1) {
    argmax_w_high = argmax_w_low = width - 1;
    argmax_w = (Dtype)argmax_w_low;
  }
  else
    argmax_w_high = argmax_w_low + 1;
  
  Dtype weight = 0;
  if (h == argmax_h_low) {
    if (w == argmax_w_low) {
      weight = (h + 1 - argmax_h) * (w + 1 - argmax_w);
    }
    else if (w == argmax_w_high) {
      weight = (h + 1 - argmax_h) * (argmax_w + 1 - w);
    }
  }
  else if (h == argmax_h_high) {
    if (w == argmax_w_low) {
      weight = (argmax_h + 1 - h) * (w + 1 - argmax_w);
    }
    else if (w == argmax_w_high) {
      weight = (argmax_h + 1 - h) * (argmax_w + 1 - w);
    }
  }
  return weight;
}
  
template <typename Dtype>
__global__ void ROIAligningBackwardFeature(const int nthreads, const Dtype* top_diff,
        const Dtype* argmax_data_h, const Dtype* argmax_data_w, const int* argmax_data, const int num_rois, const Dtype spatial_scale, const int channels,
        const int height, const int width, const int pooled_height,
        const int pooled_width, Dtype* bottom_diff, const Dtype* bottom_rois) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // (n, c, h, w) coords in bottom data
    int w = index % width;
    int h = (index / width) % height;
    int c = (index / width / height) % channels;
    int n = index / width / height / channels;
    
    Dtype gradient = 0;
    // Accumulate gradient over all ROIs that pooled this element
    for (int roi_n = 0; roi_n < num_rois; ++roi_n) {
      const Dtype* offset_bottom_rois = bottom_rois + roi_n * 5;
      int roi_level = offset_bottom_rois[0];
      // Skip if ROI's level doesn't match n
      if (n != roi_level) {
        continue;
      }
      
      Dtype dtype_roi_start_w = offset_bottom_rois[1] * spatial_scale;
      Dtype dtype_roi_start_h = offset_bottom_rois[2] * spatial_scale;
      Dtype dtype_roi_end_w = offset_bottom_rois[3] * spatial_scale;
      Dtype dtype_roi_end_h = offset_bottom_rois[4] * spatial_scale;
      int int_roi_start_w = round(offset_bottom_rois[1] * spatial_scale);
      int int_roi_start_h = round(offset_bottom_rois[2] * spatial_scale);
      int int_roi_end_w = round(offset_bottom_rois[3] * spatial_scale);
      int int_roi_end_h = round(offset_bottom_rois[4] * spatial_scale);


      // Skip if ROI doesn't include (h, w)
      const bool in_roi = (w >= floor(dtype_roi_start_w) && w <= ceil(dtype_roi_end_w) &&
         h >= floor(dtype_roi_start_h) && h <= ceil(dtype_roi_end_h));
      if (!in_roi) {
        continue;
      }

      int offset = (roi_n * channels + c) * pooled_height * pooled_width;
      const Dtype* offset_top_diff = top_diff + offset;
      const Dtype* offset_argmax_data_h = argmax_data_h + offset;
      const Dtype* offset_argmax_data_w = argmax_data_w + offset;
      const int* offset_argmax_data = argmax_data + offset;
      
      // Compute feasible set of pooled units that could have pooled
      // this bottom unit
      
      // Force malformed ROIs to be 1x1
      Dtype dtype_roi_width = max(dtype_roi_end_w - dtype_roi_start_w+(Dtype)1.0, (Dtype)1.0);
      Dtype dtype_roi_height = max(dtype_roi_end_h - dtype_roi_start_h+(Dtype)1.0, (Dtype)1.0);
      int int_roi_width = max(int_roi_end_w - int_roi_start_w + 1, 1);
      int int_roi_height = max(int_roi_end_h - int_roi_start_h + 1, 1);

      Dtype bin_size_h = static_cast<Dtype>(dtype_roi_height)  / static_cast<Dtype>(pooled_height);
      Dtype bin_size_w = static_cast<Dtype>(dtype_roi_width)  / static_cast<Dtype>(pooled_width);
      
      int int_phstart = floor(static_cast<Dtype>(h - int_roi_start_h - 1) / bin_size_h - 1);
      int int_phend = ceil(static_cast<Dtype>(h - int_roi_start_h + 1) / bin_size_h);
      int int_pwstart = floor(static_cast<Dtype>(w - int_roi_start_w - 1) / bin_size_w - 1);
      int int_pwend = ceil(static_cast<Dtype>(w - int_roi_start_w + 1) / bin_size_w);
     
      Dtype dtype_phstart = static_cast<Dtype>(h - dtype_roi_start_h - 1) / bin_size_h - 1;
      Dtype dtype_phend = static_cast<Dtype>(h - dtype_roi_start_h + 1) / bin_size_h;
      Dtype dtype_pwstart = static_cast<Dtype>(w - dtype_roi_start_w - 1) / bin_size_w - 1;
      Dtype dtype_pwend = static_cast<Dtype>(w - dtype_roi_start_w + 1) / bin_size_w;

      int_phstart = min(max(int_phstart, 0), pooled_height);
      int_phend = min(max(int_phend, 0), pooled_height);
      int_pwstart = min(max(int_pwstart, 0), pooled_width);
      int_pwend = min(max(int_pwend, 0), pooled_width);
 
      dtype_phstart = min(max(dtype_phstart , Dtype(0)), Dtype(height));
      dtype_phend = min(max(dtype_phend , Dtype(0)), Dtype(height));
      dtype_pwstart = min(max(dtype_pwstart , Dtype(0)), Dtype(width));
      dtype_pwend = min(max(dtype_pwend , Dtype(0)), Dtype(width));


      for (int ph = int_phstart; ph < int_phend; ++ph) {
        for (int pw = int_pwstart; pw < int_pwend; ++pw) {

            Dtype ih = dtype_phstart + ph - int_phstart;
            Dtype iw = dtype_pwstart + pw - int_pwstart;
    
           if(offset_argmax_data[ph*pooled_width + pw] == (h*width + w)){
               //printf("In backward for feature map\n");
               Dtype weight = get_feature_gradient(offset_argmax_data_h[ph * pooled_width + pw],
                   offset_argmax_data_w[ph * pooled_width + pw], h, w, height, width);
               gradient += weight * offset_top_diff[ph * pooled_width + pw];
           }
        }
      }
    }
    bottom_diff[index] = gradient;
  }
}
  
template <typename Dtype>
__device__ Dtype get_coordinate_gradient(int coordinate_index, Dtype h, Dtype w, 
           const Dtype* offset_bottom_data, const Dtype oh, const Dtype ow, const int height, const int width, 
           const int pooled_height, const int pooled_width) {
  
  int arg_interpolate_h = (int) h;
  int arg_interpolate_w = (int) w;
  
  if (arg_interpolate_h + 1 > height - 1 || arg_interpolate_w + 1 > width - 1) {
    return 0;
  }
  
  Dtype map_ratio_h = static_cast<Dtype>(oh) / static_cast<Dtype>(pooled_height);
  Dtype map_ratio_w = static_cast<Dtype>(ow) / static_cast<Dtype>(pooled_width);
  
  Dtype weight = 0;
  int corner_ind_1 = arg_interpolate_h * width + arg_interpolate_w;
  int corner_ind_2 = arg_interpolate_h * width + (arg_interpolate_w + 1);
  int corner_ind_3 = (arg_interpolate_h + 1) * width + arg_interpolate_w;
  int corner_ind_4 = (arg_interpolate_h + 1) * width + (arg_interpolate_w + 1);
  
  Dtype dxc = 0.0, dyc = 0.0, dw = 0.0, dh = 0.0;
  
  dxc += (-1.0 * (1.0 - h + arg_interpolate_h) * offset_bottom_data[corner_ind_1]);
  dxc += ( 1.0 * (1.0 - h + arg_interpolate_h) * offset_bottom_data[corner_ind_2]);
  dxc += (-1.0 * (h - arg_interpolate_h)       * offset_bottom_data[corner_ind_3]);
  dxc += ( 1.0 * (h - arg_interpolate_h)       * offset_bottom_data[corner_ind_4]);
  
  dyc += (-1.0 * (1.0 - w + arg_interpolate_w) * offset_bottom_data[corner_ind_1]);
  dyc += (-1.0 * (w - arg_interpolate_w)       * offset_bottom_data[corner_ind_2]);
  dyc += ( 1.0 * (1.0 - w + arg_interpolate_w) * offset_bottom_data[corner_ind_3]);
  dyc += ( 1.0 * (w - arg_interpolate_w)       * offset_bottom_data[corner_ind_4]);
  
  dw += ((0.5 - map_ratio_w) * (1.0 - h + arg_interpolate_h) * offset_bottom_data[corner_ind_1]);
  dw += ((-0.5+map_ratio_w)  * (1.0 - h + arg_interpolate_h) * offset_bottom_data[corner_ind_2]);
  dw += ((0.5- map_ratio_w)  * (h - arg_interpolate_h)       * offset_bottom_data[corner_ind_3]);
  dw += ( (-0.5+map_ratio_w) * (h - arg_interpolate_h)       * offset_bottom_data[corner_ind_4]);
  
  dh += ((0.5-map_ratio_h)   * (1.0 - w + arg_interpolate_w) * offset_bottom_data[corner_ind_1]);
  dh += ((0.5- map_ratio_h)  * ( w - arg_interpolate_w)      * offset_bottom_data[corner_ind_2]);
  dh += ( (-0.5+map_ratio_h) * (1.0 - w + arg_interpolate_w) * offset_bottom_data[corner_ind_3]);
  dh += ( (-0.5+map_ratio_h) * ( w - arg_interpolate_w)      * offset_bottom_data[corner_ind_4]);
      
  if (coordinate_index == 1) {
    // \par f / \par x1
    weight = 0.5 * dxc - dw;
  } else if (coordinate_index == 2) {
    // \par f / \par y1
    weight = 0.5 * dyc - dh;
  } else if (coordinate_index == 3) {
    // \par f / \par w
    weight = 0.5 * dxc + dw;
  } else if (coordinate_index == 4) {
    // \par f / \par h
    weight = 0.5 * dyc + dh;
  }
  return weight;
}

template <typename Dtype>
__global__ void ROIAligningBackwardCoordinate(const int nthreads, const int pooled_width, const int pooled_height, 
  const int width, const int height, const int channels, const Dtype spatial_scale, const Dtype* bottom_rois, const Dtype* bottom_data, 
  const Dtype* argmax_data_h, const Dtype* argmax_data_w, const Dtype* top_diff, Dtype* buffer_data) {
  // index is arranged as (roi_n * 5, c, w, h)
  // each element in buffer_data represents the derivative of output feature 
  // map to certain coordinate
  // coordinate_index == 0: to batch index (will always be 0)
  // coordinate_index == 1: to xc (x-center of ROI)
  // coordinate_index == 2: to yc (y-center of ROI)
  // coordinate_index == 3: to w  (width of ROI)
  // coordinate_index == 4: to h  (height of ROI)
  
  CUDA_KERNEL_LOOP(index, nthreads) {
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = (index / pooled_width / pooled_height / channels);
    int roi_n = n / 5;
    int coordinate_index = n % 5;
    Dtype gradient = 0.0;
    if (coordinate_index == 0) {
      buffer_data[index] = gradient;
    }
    
    const Dtype* offset_bottom_rois = bottom_rois + roi_n * 5;
    int roi_batch_ind = offset_bottom_rois[0];
    Dtype roi_start_w = offset_bottom_rois[1] * spatial_scale;
    Dtype roi_start_h = offset_bottom_rois[2] * spatial_scale;
    Dtype roi_end_w = offset_bottom_rois[3] * spatial_scale;
    Dtype roi_end_h = offset_bottom_rois[4] * spatial_scale;

    // Force malformed ROIs to be 1x1
    Dtype roi_width = max(roi_end_w - roi_start_w + Dtype(1), Dtype(1));
    Dtype roi_height = max(roi_end_h - roi_start_h + Dtype(1), Dtype(1));
    Dtype bin_size_h = static_cast<Dtype>(roi_height) / static_cast<Dtype>(pooled_height);
    Dtype bin_size_w = static_cast<Dtype>(roi_width)  / static_cast<Dtype>(pooled_width);
                       
    assert(roi_start_h <= roi_end_h);
    assert(roi_start_w <= roi_end_w);
    
    const Dtype* offset_bottom_data = bottom_data + ((roi_batch_ind * channels  + c) * height * width);
    
    int offset = (((roi_n * channels + c) * pooled_height + ph) * pooled_width) + pw;
    // arg max coordinate when forward
    Dtype ih = argmax_data_h[offset];
    Dtype iw = argmax_data_w[offset];
    // since we compute the max value over a set of elements during forward
    // so we re-compute the output element according to argmax_data
    // (similar for iw)
    const Dtype output_h = (ih - roi_start_h) / bin_size_h;
    const Dtype output_w = (iw - roi_start_w) / bin_size_w;
    Dtype weight = spatial_scale * get_coordinate_gradient(coordinate_index, ih, iw, offset_bottom_data, output_h, output_w, height, width, pooled_height, pooled_width);
    //printf("weight is %f\n",weight);
    //buffer_data[index] = weight * top_diff[offset];
  }
}

// used for thrust::reduce_by_key as key struct
// https://thrust.github.io/doc/group__reductions.html for more detail
template <typename T>
struct linear_index_to_row_index : public thrust::unary_function<T,T>
{
  T C; // number of columns

  __host__ __device__
  linear_index_to_row_index(T C) : C(C) {}

  __host__ __device__
  T operator()(T i) {
    return i / C;
  }
};

template <typename Dtype>
void ROIAligningLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
              const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* bottom_rois = bottom[1]->gpu_data();
  const Dtype* top_diff = top[0]->gpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  int count = bottom[0]->count();
  caffe_gpu_set(count, Dtype(0.), bottom_diff);
  const Dtype* argmax_data_h = max_idx_h_.gpu_data();
  const Dtype* argmax_data_w = max_idx_w_.gpu_data();
  const int* argmax_data = max_idx_.gpu_data();
  
  const Dtype* top_data = top[0]->gpu_data();
  // backpropgation to feature map
  if (propagate_down[0]) {
    ROIAligningBackwardFeature<Dtype> <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS >>>
      (count, top_diff, argmax_data_h, argmax_data_w, argmax_data, top[0]->num(), spatial_scale_, channels_,
       height_, width_, pooled_height_, pooled_width_, bottom_diff, bottom_rois);
  }
  
  Dtype* bottom_rois_diff = bottom[1]->mutable_gpu_diff();
  count = bottom[1]->count();
  caffe_gpu_set(count, Dtype(0.), bottom_rois_diff);
  
  // backpropgation to coordinate
  // note: for each ROI, every element of the output feature map has derivative on its coordinate
  // but it will be very slow if we aggregate all the gradient inside CUDA kernel
  // therefore we pre-computed the dirivative of coordinate for each output element (stored in buffer_)
  // and then use thrust reduce_by_key to get summation of this values 
  if (propagate_down[1]) {
    Dtype* buffer_data = buffer_.mutable_gpu_diff();
    const int buffer_count = buffer_.count();
    caffe_gpu_set(buffer_count, Dtype(0.), buffer_data);
    //ROIAligningBackwardCoordinate<Dtype><<<CAFFE_GET_BLOCKS(buffer_count), CAFFE_CUDA_NUM_THREADS>>>(
    // buffer_count, pooled_width_, pooled_height_, width_, height_, channels_,  spatial_scale_, bottom_rois, bottom_data, 
    // argmax_data_h, argmax_data_w, top_diff, buffer_data);

    // this is a standard practice for thrush::reduce_by_key
    // you may refer https://github.com/thrust/thrust/blob/master/examples/sum_rows.cu for more detail
    //int R = bottom[1]->num() * 5;
    //int C = channels_ * pooled_height_ * pooled_width_;
    //thrust::device_vector<Dtype> array(R*C);
    //thrust::copy(buffer_data, buffer_data+buffer_count, array.begin());
    //thrust::device_vector<Dtype> row_sums(R);
    //thrust::device_vector<int> row_indices(R);
    //thrust::reduce_by_key(
    //  thrust::make_transform_iterator(thrust::counting_iterator<int>(0), linear_index_to_row_index<int>(C)),
    //  thrust::make_transform_iterator(thrust::counting_iterator<int>(0), linear_index_to_row_index<int>(C)) + (R*C),
    //  array.begin(),
    //  row_indices.begin(),
    //  row_sums.begin(),
    //  thrust::equal_to<int>(),
    //  thrust::plus<Dtype>());
    //// copy back the result value to Caffe's blob
    //thrust::copy(row_sums.begin(), row_sums.end(), bottom_rois_diff);
  }
  CUDA_POST_KERNEL_CHECK;
  //LOG(INFO)<<"Backward done for roi_align";
}
  
INSTANTIATE_LAYER_GPU_FUNCS(ROIAligningLayer);
  
}  // namespace caffe

