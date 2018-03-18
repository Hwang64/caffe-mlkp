#include <cfloat>
#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/prodpara_layer.hpp"
#include "caffe/util/math_functions.hpp"
using std::ceil;

namespace caffe {

template <typename Dtype>
void ProdparaLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top){
    op_=this->layer_param_.prodpara_param().operation();
    bias_term_=this->layer_param_.prodpara_param().bias_term();
    const int num_output = this->layer_param_.prodpara_param().num_output();
    const int feat_stride = this->layer_param_.prodpara_param().feat_stride();
    const int max_height = this->layer_param_.prodpara_param().max_height();
    const int max_width  = this->layer_param_.prodpara_param().max_width();
    num_=bottom[0]->channels();
    N_=num_output;
    H_=bottom[0]->height();
    W_=bottom[0]->width();
    LOG(INFO)<<"Feature map height is "<<H_;
    LOG(INFO)<<"Feature map widht is "<<W_;
    CHECK_EQ(num_output,num_)<<"Input channels must be equal to output channels";
    if(this->blobs_.size()>0){
        LOG(INFO)<<"SKIPPING parameter initailaztion but reshape";
        this->blobs_[0]->Reshape(1,N_,63,38);
    }
    else{
        if(bias_term_){
            this->blobs_.resize(2);
        }
        else{
            this->blobs_.resize(1);
        }
        vector<int> weight_shape(4);
        switch(op_){
            case ProdparaParameter_ProdparaOp_ACROSS_FEATURE_MAP:
                LOG(INFO)<<"Selection across feature map";
                scale_.Reshape(1,1,H_,W_);
                weight_shape[0]=top[0]->num();
                weight_shape[1]=1;
                weight_shape[2]=static_cast<int>(ceil(static_cast<int>(max_height) / feat_stride))+1;
                weight_shape[3]=static_cast<int>(ceil(static_cast<int>(max_width) / feat_stride))+1;
                LOG(INFO)<<"shape_2 is"<<weight_shape[2];
                LOG(INFO)<<"shape_3 is"<<weight_shape[3];
                break;
            case ProdparaParameter_ProdparaOp_ACROSS_CHANNEL:
                LOG(INFO)<<"Selection across channel";
                scale_.Reshape(1,N_,1,1);
                weight_shape[0]=top[0]->num();
                weight_shape[1]=N_;
                weight_shape[2]=1;
                weight_shape[3]=1;
                break;
            case ProdparaParameter_ProdparaOp_ACROSS_ALL:
                LOG(INFO)<<"Selection across ALL";
                scale_.Reshape(1,N_,H_,W_);
                weight_shape[0]=top[0]->num();
                weight_shape[1]=N_;
                weight_shape[2]=static_cast<int>(ceil(static_cast<float>(max_height) / feat_stride))+1;
                weight_shape[3]=static_cast<int>(ceil(static_cast<float>(max_width) / feat_stride))+1;
                break;
            default:
                LOG(FATAL) << "Unknown operation";
        }
        scale_count=scale_.count();
        //LOG(INFO)<<"Scale count is "<<scale_count;
        caffe_set(scale_count,Dtype(0),scale_.mutable_cpu_data());
        this->blobs_[0].reset(new Blob<Dtype>(weight_shape));
        weight_count=this->blobs_[0]->count();
        // fill the weights
        shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
            this->layer_param_.prodpara_param().weight_filler()));
        weight_filler->Fill(this->blobs_[0].get());
        // If necessary, intiialize and fill the bias term
        if (bias_term_) {
            vector<int> bias_shape(1, N_);
            this->blobs_[1].reset(new Blob<Dtype>(bias_shape));
            shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
                this->layer_param_.prodpara_param().bias_filler()));
            bias_filler->Fill(this->blobs_[1].get());
        }
        this->param_propagate_down_.resize(this->blobs_.size(), true);
        LOG(INFO)<<"Set-up done";
    }
}

template <typename Dtype>
void ProdparaLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top){
    for (int i=0;i<bottom.size();++i){
        CHECK(bottom[i]->shape()==bottom[0]->shape());
    }
    top[0]->ReshapeLike(*bottom[0]);
    CHECK(top[0]->shape()==bottom[0]->shape());
}

template <typename Dtype>
void ProdparaLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom, 
        const vector<Blob<Dtype>*>& top){

    const int count_num=top[0]->count();
    N_=bottom[0]->shape(1);
    H_=bottom[0]->shape(2);
    W_=bottom[0]->shape(3);
    Dtype* top_data=top[0]->mutable_cpu_data();
    const Dtype* bottom_data=bottom[0]->cpu_data();
    Dtype* weight=this->blobs_[0]->mutable_cpu_data();
    const Dtype* bias=this->blobs_[1]->cpu_data();
    Blob<Dtype> buffer_spatial_;
    buffer_spatial_.Reshape(bottom[0]->num(),1,bottom[0]->height(),bottom[0]->width());
    Blob<Dtype> buffer_weight_;
    buffer_weight_.Reshape(bottom[0]->num(),bottom[0]->channels(),bottom[0]->height(),bottom[0]->width());
    Dtype* buffer_spatial=buffer_spatial_.mutable_cpu_data();
    Dtype* buffer_weight=buffer_weight_.mutable_cpu_data();
    caffe_set(H_*W_,Dtype(1),buffer_spatial);    
    CHECK(top[0]->shape()==bottom[0]->shape());
    switch(op_){
    case ProdparaParameter_ProdparaOp_ACROSS_FEATURE_MAP:
            for(int i=0;i<N_;++i){
                caffe_mul(weight_count,weight,bottom_data,top_data);
                bottom_data+=bottom[0]->offset(0,1,0,0);
                top_data+=top[0]->offset(0,1,0,0);
            }
            break;
    case ProdparaParameter_ProdparaOp_ACROSS_CHANNEL:
            LOG(INFO)<<"debug7";
            caffe_cpu_gemm<Dtype>(CblasNoTrans,CblasNoTrans,N_,H_*W_,1,Dtype(1),weight,buffer_spatial,Dtype(0),buffer_weight);
            LOG(INFO)<<"debug8";
            caffe_mul<Dtype>(count_num,buffer_weight,bottom_data,top_data);
            LOG(INFO)<<"debug9";
        break;
    default:
        LOG(FATAL)<<"Unknown operation";
    }
    if(bias_term_){
        bias =this->blobs_[1]->mutable_cpu_data();
        caffe_add(count_num,bias,bottom[0]->cpu_data(),top_data);
    }



}

template <typename Dtype>
void ProdparaLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& bottom,
        const vector<bool>& propagate_down,const vector<Blob<Dtype>*>& top){

    const int count_num=top[0]->count();
    int NUM_=bottom[0]->shape(0);
    N_=bottom[0]->shape(1);
    H_=bottom[0]->shape(2);
    W_=bottom[0]->shape(3);
    const Dtype* top_diff=top[0]->cpu_diff();
    Dtype* top_data=top[0]->mutable_cpu_data();
    Dtype* bottom_diff=bottom[0]->mutable_cpu_diff();
    //const Dtype* bottom_data=bottom[0]->cpu_data();
    Dtype* weight=this->blobs_[0]->mutable_cpu_data();
    Dtype* weight_diff=this->blobs_[0]->mutable_cpu_diff();
    //const Dtype* bias=this->blobs_[1]->cpu_data();
    Blob<Dtype> buffer_spatial_;
    buffer_spatial_.Reshape(bottom[0]->num(),1,bottom[0]->height(),bottom[0]->width());
    Blob<Dtype> buffer_;
    buffer_.Reshape(1, bottom[0]->channels(),bottom[0]->height(), bottom[0]->width());
    Blob<Dtype> buffer_weight_;
    buffer_weight_.Reshape(bottom[0]->num(),bottom[0]->channels(),bottom[0]->height(),bottom[0]->width());
    Blob<Dtype> buffer_channel_;
    buffer_channel_.Reshape(1, bottom[0]->channels(), 1, 1);
    Dtype* buffer_weight=buffer_weight_.mutable_cpu_data();
    Dtype* buffer_spatial=buffer_spatial_.mutable_cpu_data();
    Dtype* buffer_data = buffer_.mutable_cpu_data();
    Dtype* buffer_channel = buffer_channel_.mutable_cpu_data();
    caffe_set(H_*W_,Dtype(1),buffer_spatial);
    int dim=count_num / NUM_;

    switch(op_){
    case ProdparaParameter_ProdparaOp_ACROSS_CHANNEL:
        LOG(INFO)<<"calculate weight diff";
        for(int n=0;n<NUM_;n++){
            LOG(INFO)<<"debug10";
            caffe_mul<Dtype>(dim, top_data+n*dim, top_diff+n*dim, buffer_data);
            LOG(INFO)<<"debug11";
            caffe_cpu_gemv<Dtype>(CblasNoTrans, N_, H_*W_, Dtype(1),
                                buffer_data, buffer_spatial, Dtype(0),
                                buffer_channel);
            caffe_div<Dtype>(N_, buffer_channel, weight, buffer_channel);
            LOG(INFO)<<"debug12";
            caffe_add<Dtype>(N_, buffer_channel, weight_diff, weight_diff);
        }
        LOG(INFO)<<"calculate blob diff";
        caffe_cpu_gemm<Dtype>(CblasNoTrans,CblasNoTrans,N_,H_*W_,1,Dtype(1),weight,buffer_spatial,Dtype(0),buffer_weight);
        LOG(INFO)<<"debug13";
        caffe_mul<Dtype>(count_num,buffer_weight,top_diff,bottom_diff);
        LOG(INFO)<<"debug4";
        break;
    case ProdparaParameter_ProdparaOp_ACROSS_FEATURE_MAP:
        //LOG(INFO)<<"debug10";
        for(int i=0;i<N_;i++){
            caffe_copy(weight_count,weight,bottom_diff);
            bottom_diff+=bottom[0]->offset(0,1,0,0);
        }
        //LOG(INFO)<<"debug11";
        break;
    default:
        LOG(FATAL) <<"Unknown operation";
    }

}
#ifdef CPU_ONLY
STUB_cpu(ProdparaLayer);
#endif

INSTANTIATE_CLASS(ProdparaLayer);
REGISTER_LAYER_CLASS(Prodpara);

} 
