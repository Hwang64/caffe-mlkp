#include <cfloat>
#include <vector>

#include "caffe/layers/locaware_layer.hpp"
#include "caffe/util/math_functions.hpp"
using std::ceil;

namespace caffe {

template<typename Dtype>
void LocawareLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top){
    
    op_=this->layer_param_.locaware_param().operation();
    const int channel_num = this->layer_param_.locaware_param().channel_num();
    const int width_num = this->layer_param_.locaware_param().width();
    const int height_num = this->layer_param_.locaware_param().height();  
    const int bottom_0_num = bottom[0]->num();
    const int bottom_1_num = bottom[1]->num();
    Num_ = top[0]->count();
    N_ = bottom[0]->num();
    C_ = bottom[0]->channels();
    H_ = bottom[1]->height();
    W_ = bottom[1]->width();
    CHECK_EQ(channel_num,C_)<<"CHANNEL_NUM EQUAL ERROR";
    CHECK_EQ(height_num,H_)<<"HEIGHT_NUM EQUAL ERROR";
    CHECK_EQ(width_num,W_)<<"WIDTH_NUM EQUAL ERROR";
    CHECK_EQ(bottom_0_num,bottom_1_num)<<"BATCHSIZE_NUM EQUAL ERROR";
}

template<typename Dtype>
void LocawareLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top){
    top[0]->Reshape(N_,C_,H_,W_);
}

template<typename Dtype>
void LocawareLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top){

}

template<typename Dtype>
void LocawareLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
        const vector<Blob<Dtype>*>& bottom){

}

#ifdef CPU_ONLY
STUB_cpu(LocawareLayer);
#endif

INSTANTIATE_CLASS(LocawareLayer);
REGISTER_LAYER_CLASS(Locaware);

}
