#ifndef CAFFE_L2Norm_LAYERS_HPP_
#define CAFFE_L2Norm_LAYERS_HPP_

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"


namespace caffe{

  template <typename Dtype>
  class L2NormalizationLayer : public Layer<Dtype>{
  public:
  	explicit L2NormalizationLayer(const LayerParameter& param)
  		: Layer<Dtype>(param){}
  	virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
  		const vector<Blob<Dtype>*>& top);
  	virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
  		const vector<Blob<Dtype>*>& top);

  	virtual inline const char* type() const { return "L2Normalization"; }
  	virtual inline int ExactNumBottomBlobs() const { return 1; }
  	virtual inline int ExactNumTopBlobs() const { return 1; }

  protected:
  	virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
  		const vector<Blob<Dtype>*>& top);
  	virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
  		const vector<Blob<Dtype>*>& top);
  	virtual void Backward_cpu(const vector<Blob<Dtype>*>&top,
  		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  	virtual void Backward_gpu(const vector<Blob<Dtype>*>&top,
  		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  	virtual void ChannelForward_cpu(const vector<Blob<Dtype>*>& bottom,
  		const vector<Blob<Dtype>*>& top);
  	virtual void ChannelForward_gpu(const vector<Blob<Dtype>*>& bottom,
  		const vector<Blob<Dtype>*>& top);
    virtual void AllForward_cpu(const vector<Blob<Dtype>*>& bottom,
    	const vector<Blob<Dtype>*>& top);
  	virtual void AllForward_gpu(const vector<Blob<Dtype>*>& bottom,
  		const vector<Blob<Dtype>*>& top);
  	virtual void ChannelBackward_cpu(const vector<Blob<Dtype>*>& bottom,
  		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& top);
  	virtual void ChannelBackward_gpu(const vector<Blob<Dtype>*>& bottom,
  		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& top);
  	virtual void AllBackward_cpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
  		const vector<Blob<Dtype>*>& bottom);
    virtual void AllBackward_gpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
  		const vector<Blob<Dtype>*>& bottom);

  	int size_;
  	int pre_pad_;
  	Dtype k_;
  	Dtype beta_;
  	int num_;
  	int channels_;
  	int height_;
  	int width_;

  	Blob<Dtype> scale_;
  };

}// end of caffe namespace
#endif
