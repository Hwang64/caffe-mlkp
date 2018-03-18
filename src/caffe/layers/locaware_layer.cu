#include <cfloat>
#include <vector>

#include "caffe/layers/locaware_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe{

template <typename Dtype>
void LocawareLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top){

    Num_ = top[0]->count();
    N_ = bottom[0]->num();
    C_ = bottom[0]->channels();
    H_ = bottom[1]->height();
    W_ = bottom[1]->width();
    top[0]->Reshape(N_,C_,H_,W_);
    const int top_shape = top[0]->count();
    int spatial_dim = H_*W_;
    int channel_dim = C_;
    Dtype* top_data = top[0]->mutable_gpu_data();
    Dtype* buffer_spatial;
    Dtype* buffer_channel;
    Dtype* buffer_data;
    const Dtype* bottom0_data = bottom[0]->gpu_data();
    const Dtype* bottom1_data = bottom[1]->gpu_data();
    Blob<Dtype> buffer_spatial_;
    Blob<Dtype> buffer_channel_;
    Blob<Dtype> buffer_data_;
    buffer_spatial_.Reshape(top[0]->num(),1,H_,W_);
    buffer_channel_.Reshape(top[0]->num(),C_,1,1);
    buffer_data_.Reshape(top[0]->num(),C_,H_,W_);
    caffe_gpu_set<Dtype>(spatial_dim,Dtype(1),buffer_spatial_.mutable_gpu_data());    
    caffe_gpu_set<Dtype>(channel_dim,Dtype(1),buffer_channel_.mutable_gpu_data());    
    buffer_spatial=buffer_spatial_.mutable_gpu_data();
    buffer_channel=buffer_channel_.mutable_gpu_data();
    switch(op_){
        case LocAwareParameter_LocAwareOp_DIRECTLY_MAPPING:
            caffe_gpu_gemm<Dtype>(CblasNoTrans,CblasNoTrans,C_,H_*W_,1,Dtype(1),bottom0_data,bottom1_data,Dtype(0),top_data);
            //buffer_data=buffer_data_.mutable_gpu_data();
            //caffe_gpu_add<Dtype>(top_shape,buffer_data,top_data,top_data);
            break;
        case LocAwareParameter_LocAwareOp_IDENTITY_MAPPING:
            caffe_gpu_gemm<Dtype>(CblasNoTrans,CblasNoTrans,C_,W_*H_,1,Dtype(1),bottom0_data,buffer_spatial,Dtype(0),top_data);
            break;
        default:
            LOG(FATAL)<<"Unkown Operation";
    }
}

template <typename Dtype>
void LocawareLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
         const vector<Blob<Dtype>*>& bottom){

    Num_ = top[0]->count();
    N_ = bottom[0]->num();
    C_ = bottom[0]->channels();
    H_ = bottom[1]->height();
    W_ = bottom[1]->width();
    top[0]->Reshape(N_,C_,H_,W_);
    const int top_shape = top[0]->count();
    int spatial_dim = H_*W_;
    int channel_dim = C_;
    const Dtype* top_data = top[0]->gpu_data();
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* buffer_spatial;
    Dtype* buffer_channel;
    Dtype* buffer_data;
    Dtype* bottom0_diff = bottom[0]->mutable_gpu_diff();
    Dtype* bottom1_diff = bottom[1]->mutable_gpu_diff();
    const Dtype* bottom0_data = bottom[0]->gpu_data();
    const Dtype* bottom1_data = bottom[1]->gpu_data();
    Blob<Dtype> buffer_spatial_;
    Blob<Dtype> buffer_channel_;
    Blob<Dtype> buffer_data_;
    buffer_spatial_.Reshape(top[0]->num(),1,H_,W_);
    buffer_channel_.Reshape(top[0]->num(),C_,1,1);
    buffer_data_.Reshape(top[0]->num(),C_,H_,W_);
    caffe_gpu_set<Dtype>(spatial_dim,Dtype(1),buffer_spatial_.mutable_gpu_data());    
    caffe_gpu_set<Dtype>(channel_dim,Dtype(1),buffer_channel_.mutable_gpu_data());    
    buffer_spatial=buffer_spatial_.mutable_gpu_data();
    buffer_channel=buffer_channel_.mutable_gpu_data();
    switch(op_){
    case LocAwareParameter_LocAwareOp_DIRECTLY_MAPPING:
        caffe_gpu_gemv<Dtype>(CblasNoTrans,C_,H_*W_,Dtype(1),top_diff,bottom1_data,Dtype(0),bottom0_diff);
        caffe_gpu_gemv<Dtype>(CblasTrans,C_,H_*W_,Dtype(1),top_diff,bottom0_data,Dtype(0),bottom1_diff);
    case LocAwareParameter_LocAwareOp_IDENTITY_MAPPING:
        caffe_gpu_gemv<Dtype>(CblasNoTrans,C_,H_*W_,Dtype(1),top_diff,buffer_spatial,Dtype(0),bottom0_diff);
        break;
    default:
        LOG(FATAL)<<"Unknown Operation";
    }

}

INSTANTIATE_LAYER_GPU_FUNCS(LocawareLayer);
}
