#ifndef CAFFE_PRODPARA_LAYER_HPP_
#define CAFFE_PRODPARA_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
class LocawareLayer : public Layer<Dtype>{

 public:
  explicit LocawareLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "Prodpara"; }
  virtual inline int MinBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  LocAwareParameter_LocAwareOp op_;
  int Num_;
  int N_;
  int C_;
  int H_;
  int W_;
  Blob<Dtype> Channel_;
  Blob<Dtype> Feature_;
  Dtype* channel_data;
  Dtype* feature_data;
  int channel_size;
  int feature_size;
};

} // namespace caffe

#endif // CAFFE_LOCAWARE_LAYER_HPP_
