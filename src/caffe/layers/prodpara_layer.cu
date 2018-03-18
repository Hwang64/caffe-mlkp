#include <cfloat>
#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/prodpara_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void ProdparaLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, 
        const vector<Blob<Dtype>*>& top){

    const int count_num=top[0]->count();
    int NUM_=bottom[0]->shape(0);
    N_=bottom[0]->shape(1);
    H_=bottom[0]->shape(2);
    W_=bottom[0]->shape(3);
    Dtype* top_data=top[0]->mutable_gpu_data();
    const Dtype* bottom_data=bottom[0]->gpu_data();
    Dtype* weight=this->blobs_[0]->mutable_gpu_data();
    const Dtype* bias;
    int spatial_dim=H_*W_;
    int channel_dim=N_;
    int weight_height=this->blobs_[0]->shape(2);
    int weight_width=this->blobs_[0]->shape(3);
    Blob<Dtype> buffer_spatial_;
    buffer_spatial_.Reshape(bottom[0]->num(),1,bottom[0]->height(),bottom[0]->width());
    Blob<Dtype> buffer_channel_;
    buffer_channel_.Reshape(bottom[0]->num(),bottom[0]->channels(),1,1);
    Blob<Dtype> buffer_weight_;
    buffer_weight_.Reshape(bottom[0]->num(),bottom[0]->channels(),bottom[0]->height(),bottom[0]->width());
    Dtype* buffer_spatial;
    Dtype* buffer_weight;
    Dtype* buffer_channel; 
    //Dtype* buffer_blob;
    caffe_gpu_set<Dtype>(spatial_dim,Dtype(1),buffer_spatial_.mutable_gpu_data());    
    caffe_gpu_set<Dtype>(channel_dim,Dtype(1),buffer_channel_.mutable_gpu_data());    
    buffer_spatial=buffer_spatial_.mutable_gpu_data();
    buffer_channel=buffer_channel_.mutable_gpu_data();
    CHECK(top[0]->shape()==bottom[0]->shape());
    switch(op_){
        case ProdparaParameter_ProdparaOp_ACROSS_CHANNEL:
            caffe_gpu_gemm<Dtype>(CblasNoTrans,CblasNoTrans,N_,H_*W_,1,Dtype(1),weight,buffer_spatial,Dtype(0),buffer_weight_.mutable_gpu_data());
            buffer_weight=buffer_weight_.mutable_gpu_data();
            caffe_gpu_mul<Dtype>(count_num,buffer_weight,bottom_data,top_data);
            break;
        case ProdparaParameter_ProdparaOp_ACROSS_FEATURE_MAP:
            //LOG(INFO)<<"calculate forward";
            this->blobs_[0]->Reshape(bottom[0]->num(),1,bottom[0]->height(),bottom[0]->width());
            weight=this->blobs_[0]->mutable_gpu_data();
            caffe_gpu_gemm<Dtype>(CblasNoTrans,CblasNoTrans,N_,H_*W_,1,Dtype(1),buffer_channel,weight,Dtype(0),buffer_weight_.mutable_gpu_data());
            buffer_weight=buffer_weight_.mutable_gpu_data();
            caffe_gpu_mul<Dtype>(count_num,buffer_weight,bottom_data,top_data);
            this->blobs_[0]->Reshape(bottom[0]->num(),1,weight_height,weight_width);
            break;
        case ProdparaParameter_ProdparaOp_ACROSS_ALL:
            this->blobs_[0]->Reshape(bottom[0]->num(),bottom[0]->channels(),bottom[0]->height(),bottom[0]->width());
            weight=this->blobs_[0]->mutable_gpu_data();
            caffe_gpu_mul<Dtype>(count_num,weight,bottom_data,top_data);
            this->blobs_[0]->Reshape(bottom[0]->num(),bottom[0]->channels(),weight_height,weight_width);
            break;
    default:
        LOG(FATAL)<<"Unknown operation";
    }
    if(bias_term_){
        bias =this->blobs_[1]->mutable_gpu_data();
        caffe_gpu_add(count_num,bias,bottom[0]->gpu_data(),top_data);
    }
}

template <typename Dtype>
void ProdparaLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& bottom,
        const vector<bool>& propagate_down,const vector<Blob<Dtype>*>& top){

    const int count_num=top[0]->count();
    int NUM_=bottom[0]->shape(0);
    N_=bottom[0]->shape(1);
    H_=bottom[0]->shape(2);
    W_=bottom[0]->shape(3);
    const Dtype* top_diff=top[0]->gpu_diff();
    Dtype* top_data=top[0]->mutable_gpu_data();
    Dtype* bottom_diff=bottom[0]->mutable_gpu_diff();
    const Dtype* bottom_data=bottom[0]->gpu_data();
    Dtype* weight=this->blobs_[0]->mutable_gpu_data();
    Dtype* weight_diff=this->blobs_[0]->mutable_gpu_diff();
    Blob<Dtype> buffer_spatial_;
    buffer_spatial_.Reshape(bottom[0]->num(),1,bottom[0]->height(),bottom[0]->width());
    Blob<Dtype> buffer_;
    buffer_.Reshape(1, bottom[0]->channels(),bottom[0]->height(), bottom[0]->width());
    Blob<Dtype> buffer_weight_;
    buffer_weight_.Reshape(bottom[0]->num(),bottom[0]->channels(),bottom[0]->height(),bottom[0]->width());
    Blob<Dtype> buffer_channel_;
    buffer_channel_.Reshape(1, bottom[0]->channels(), 1, 1);
    Dtype* buffer_weight;
    Dtype* buffer_spatial;
    Dtype* buffer_data;
    Dtype* buffer_channel;
    int dim=count_num / NUM_;
    int weight_height=this->blobs_[0]->shape(2);
    int weight_width=this->blobs_[0]->shape(3);
    //Dtype* blob_diff_buffer;

    switch(op_){
    case ProdparaParameter_ProdparaOp_ACROSS_CHANNEL:
        //LOG(INFO)<<"calculate weight diff";
        for(int n=0;n<NUM_;n++){
            caffe_gpu_set<Dtype>(H_*W_,Dtype(1),buffer_spatial_.mutable_gpu_data());
            buffer_spatial=buffer_spatial_.mutable_gpu_data();
            caffe_gpu_mul<Dtype>(dim, top_data+n*dim, top_diff+n*dim, buffer_.mutable_gpu_data());
            buffer_data = buffer_.mutable_gpu_data();
            caffe_gpu_gemv<Dtype>(CblasNoTrans,N_,H_*W_,Dtype(1),buffer_data,buffer_spatial,Dtype(0),buffer_channel_.mutable_gpu_data());
            buffer_channel = buffer_channel_.mutable_gpu_data();
            caffe_gpu_div<Dtype>(N_, buffer_channel, weight, buffer_channel);
            caffe_gpu_add<Dtype>(N_, buffer_channel, weight_diff, weight_diff);
        }
        //LOG(INFO)<<"calculate blob diff";
        caffe_gpu_gemm<Dtype>(CblasNoTrans,CblasNoTrans,N_,H_*W_,1,Dtype(1),weight,buffer_spatial,Dtype(0),buffer_weight_.mutable_gpu_data());
        buffer_weight=buffer_weight_.mutable_gpu_data();
        caffe_gpu_mul<Dtype>(count_num,buffer_weight,top_diff,bottom_diff);
        break;
    case ProdparaParameter_ProdparaOp_ACROSS_FEATURE_MAP:
        //LOG(INFO)<<"calculate weight diff";
        for(int n=0;n<NUM_;n++){
            caffe_gpu_set<Dtype>(N_,Dtype(1),buffer_channel_.mutable_gpu_data());
            buffer_channel=buffer_channel_.mutable_gpu_data();
            caffe_gpu_mul<Dtype>(dim,top_data+n*dim,top_diff+n*dim,buffer_.mutable_gpu_data());
            buffer_data = buffer_.mutable_gpu_data();
            caffe_gpu_gemv<Dtype>(CblasNoTrans,N_,H_*W_,Dtype(1),buffer_data,buffer_channel,Dtype(0),buffer_spatial_.mutable_gpu_data());
            buffer_spatial=buffer_spatial_.mutable_gpu_data();
            this->blobs_[0]->Reshape(bottom[0]->num(),1,bottom[0]->height(),bottom[0]->width());
            weight=this->blobs_[0]->mutable_gpu_data();
            weight_diff=this->blobs_[0]->mutable_gpu_diff();
            caffe_gpu_div<Dtype>(H_*W_,buffer_spatial,weight,buffer_spatial);
            caffe_gpu_add<Dtype>(H_*W_,buffer_spatial,weight_diff,weight_diff);
        }
        //LOG(INFO)<<"calculate blob diff";
        caffe_gpu_gemm<Dtype>(CblasNoTrans,CblasNoTrans,N_,H_*W_,1,Dtype(1),buffer_channel,weight,Dtype(0),buffer_weight_.mutable_gpu_data());
        buffer_weight=buffer_weight_.mutable_gpu_data();
        caffe_gpu_mul<Dtype>(count_num,buffer_weight,top_diff,bottom_diff);
        this->blobs_[0]->Reshape(bottom[0]->num(),1,weight_height,weight_width);
        break;
    case ProdparaParameter_ProdparaOp_ACROSS_ALL:
        //LOG(INFO)<<"calculate weight diff";
        for(int n=0;n<NUM_;n++){
            caffe_gpu_mul<Dtype>(dim,top_data+n*dim,top_diff+n*dim,buffer_.mutable_gpu_data());
            buffer_data = buffer_.mutable_gpu_data();
            this->blobs_[0]->Reshape(bottom[0]->num(),bottom[0]->channels(),bottom[0]->height(),bottom[0]->width());
            weight=this->blobs_[0]->mutable_gpu_data();
            caffe_gpu_div<Dtype>(count_num,buffer_data,weight,buffer_data);
            caffe_gpu_add<Dtype>(count_num,buffer_data,weight_diff,weight_diff);
        }
        //LOG(INFO)<<"calculate blob diff";
        caffe_gpu_mul<Dtype>(count_num,weight,top_diff,bottom_diff);
        this->blobs_[0]->Reshape(bottom[0]->num(),bottom[0]->channels(),weight_height,weight_width);
        break;
    default:
        LOG(FATAL) <<"Unknown operation";
    }
}

INSTANTIATE_LAYER_GPU_FUNCS(ProdparaLayer);

}
