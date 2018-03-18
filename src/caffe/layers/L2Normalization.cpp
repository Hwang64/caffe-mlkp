#include <vector>

#include "caffe/util/math_functions.hpp"
#include "caffe/layers/L2Norm.hpp"

namespace caffe {

	template <typename Dtype>
	void L2NormalizationLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		beta_ = this->layer_param_.l2n_param().beta();
		k_ = this->layer_param_.l2n_param().k();
	}

	template <typename Dtype>
	void L2NormalizationLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		CHECK_EQ(4, bottom[0]->num_axes()) << "Input must have 4 axes, "
			<< "corresponding to (num, channels, height, width)";
		num_ = bottom[0]->num();
		channels_ = bottom[0]->channels();
		height_ = bottom[0]->height();
		width_ = bottom[0]->width();
		switch (this->layer_param_.l2n_param().norm_region()) {
		case L2Parameter_NormRegion_ACROSS_CHANNELS:
			top[0]->Reshape(num_, channels_, height_, width_);
			scale_.Reshape(num_, channels_, height_, width_);
			break;
		case L2Parameter_NormRegion_ALL_ENTRY:
			top[0]->Reshape(num_,channels_,height_,width_);
			scale_.Reshape(num_,1,1,1);
			break;
		}
	}

	template <typename Dtype>
	void L2NormalizationLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		switch (this->layer_param_.l2n_param().norm_region()) {
		case L2Parameter_NormRegion_ACROSS_CHANNELS:
			ChannelForward_cpu(bottom, top);
			break;
		case L2Parameter_NormRegion_ALL_ENTRY:
			AllForward_cpu(bottom, top);
			break;
		default:
			LOG(FATAL) << "Unknown normalization region.";
		}
	}

	template <typename Dtype>
	void L2NormalizationLayer<Dtype>::ChannelForward_cpu(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
		const Dtype* bottom_data = bottom[0]->cpu_data();
		Dtype* top_data = top[0]->mutable_cpu_data();
		Dtype* scale_data = scale_.mutable_cpu_data();
		//top_data���������ݣ���bottom���������ݡ�
		// start with the constant value,����Ϊ1e-10�Ϳ���,������Ϊ0��Ϊ�˷�ֹ��0��������������
		for (int i = 0; i < scale_.count(); ++i) {
			scale_data[i] = Dtype(1e-10);
		}
		Blob<Dtype> square(1, channels_, height_, width_);
		Dtype* square_data = square.mutable_cpu_data();
		caffe_set(square.count(), Dtype(0), square_data);
		//Dtype alpha = alpha_;//������û��oversize��˵������������ֱ�Ӹ�Ϊalpha���������ǿ�����һ��������һЩ
		                     //���򻯷�ʽ��
		// go through the images
		for (int n = 0; n < num_; ++n) {
			// compute the padded square
			caffe_sqr(channels_ * height_ * width_,
				bottom_data + bottom[0]->offset(n),
				square_data);//����ֻ��Ҫд��data�ĳ�ʼλ�ü��ɡ�
						     //�൱��3D��������Ԫ��
	                         //ȫ��ƽ��
			//���ǰ�Channel���͡�

			for (int c = 0; c < channels_; ++c) {
				caffe_axpy<Dtype>(height_ * width_, Dtype(1),
					square_data + square.offset(0, c),
					scale_data+scale_.offset(n,0));
			}
			for (int c = 1; c < channels_; ++c) {
				caffe_copy<Dtype>(height_ * width_,
					scale_data + scale_.offset(n, 0),
					scale_data + scale_.offset(n, c));
			}
		}

		// In the end, compute output
		caffe_powx<Dtype>(scale_.count(), scale_data, -beta_, top_data); //betaΪ1/2�Ϳ����ˡ�
		caffe_mul<Dtype>(scale_.count(), top_data, bottom_data, top_data);
	}

	template <typename Dtype>
	void L2NormalizationLayer<Dtype>::AllForward_cpu(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
			const Dtype* bottom_data = bottom[0]->cpu_data();
			Dtype* top_data = top[0]->mutable_cpu_data();
			//Dtype* scale_data = scale_.mutable_cpu_data();
			Blob<Dtype> square(1, channels_, height_, width_);
  		Dtype* square_data = square.mutable_cpu_data();
  		caffe_set(square.count(), Dtype(0), square_data);
			for (int n = 0; n < num_; ++n) {
  			// compute the padded square
  			caffe_sqr(channels_ * height_ * width_,
  				bottom_data + bottom[0]->offset(n),
  				square_data);
	  		 Dtype normsqr = caffe_cpu_asum<Dtype>(channels_ * height_ * width_,
				 square_data);
	  	   caffe_cpu_scale<Dtype>(channels_ * height_ * width_,pow(normsqr,-beta_),
				 bottom_data+bottom[0]->offset(n),top_data+top[0]->offset(n));
	  	}
	}

	template <typename Dtype>
	void L2NormalizationLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
		switch (this->layer_param_.l2n_param().norm_region()) {
		case L2Parameter_NormRegion_ACROSS_CHANNELS:
			ChannelBackward_cpu(top, propagate_down, bottom);
			break;
		case L2Parameter_NormRegion_ALL_ENTRY:
			AllBackward_cpu(top, propagate_down, bottom);
			break;
		default:
			LOG(FATAL) << "Unknown normalization region.";
		}
	}

	template <typename Dtype>
	void L2NormalizationLayer<Dtype>::ChannelBackward_cpu(
		const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
		const vector<Blob<Dtype>*>& bottom) {
		//���򴫲������������ݺ��������ݵ�λ�õߵ���
		const Dtype* top_diff = top[0]->cpu_diff();//���ֻ��ߵ���
		const Dtype* top_data = top[0]->cpu_data();
		//const Dtype* bottom_data = bottom[0]->cpu_data();
		const Dtype* scale_data = scale_.cpu_data();
		Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
		Blob<Dtype> norm_ratio(1, channels_, height_, width_);//Blob<Dtype>ָ�����������͵�
		                                                      //Dtype��
		Blob<Dtype> accum_ratio(1, 1, height_, width_);
		Dtype* norm_ratio_data = norm_ratio.mutable_cpu_data();
		Dtype* accum_ratio_data = accum_ratio.mutable_cpu_data();
		// We hack a little bit by using the diff() to store an additional result
		//Dtype* accum_ratio_times_bottom = accum_ratio.mutable_cpu_diff();
		caffe_set(norm_ratio.count(), Dtype(0), norm_ratio_data);
		//Dtype cache_ratio_value = 2. * alpha_ * beta_ / size_;
		// go through individual data
		Blob<Dtype> temp(num_, channels_, height_, width_);
		//Blob<Dtype> b(num_, channels_, height_, width_);
		Dtype* temp_data = temp.mutable_cpu_data();
		for (int n = 0; n < num_; ++n) {
			int block_offset = scale_.offset(n);
			// first, compute diff_i * y_i / s_i
			caffe_mul<Dtype>(channels_*height_*width_,
				top_diff + block_offset, top_data + block_offset,
				norm_ratio_data);//����diff_i*y_i / s_i;
			caffe_set(accum_ratio.count(), Dtype(0), accum_ratio_data);
			for (int c = 0; c < channels_; ++c){
				//int acc_offset = accum_ratio.offset(0, c);
				int norm_ratio_offset = norm_ratio.offset(0, c);
				caffe_axpy<Dtype>(height_*width_,1., norm_ratio_data + norm_ratio_offset,
					accum_ratio_data);//��diff_i*y_i / s_i����ͨ�����͡�
			}
			for (int c = 0; c < channels_; ++c){
				int top_offset = top[0]->offset(n, c);
				caffe_mul<Dtype>(height_*width_, top_data + top_offset,
					accum_ratio_data,temp_data+top_offset);//�����ͺ���ͨ����ʽ����top_data��
														  //��ǰ�򴫲�ʱ��������
			}
			caffe_sub<Dtype>(channels_*height_*width_, top_diff + top[0]->offset(n),
				temp_data + temp.offset(n), temp_data+temp.offset(n));
		}
		caffe_powx<Dtype>(scale_.count(), scale_data, -beta_, bottom_diff);
		caffe_mul<Dtype>(scale_.count(), temp_data, bottom_diff, bottom_diff);
	}

	template <typename Dtype>
	void L2NormalizationLayer<Dtype>::AllBackward_cpu(
		const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
		const vector<Blob<Dtype>*>& bottom) {
      const Dtype* top_diff = top[0]->cpu_diff();
			const Dtype* top_data = top[0]->cpu_data();
			const Dtype* bottom_data = bottom[0]->cpu_data();
			Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
			int n = top[0]->num();
			int d = top[0]->count() / n;
			for (int i=0; i<n; ++i) {
      Dtype a = caffe_cpu_dot(d, top_data+i*d, top_diff+i*d);
      caffe_cpu_scale(d, a, top_data+i*d, bottom_diff+i*d);
      caffe_sub(d, top_diff+i*d, bottom_diff+i*d, bottom_diff+i*d);
      a = caffe_cpu_dot(d, bottom_data+i*d, bottom_data+i*d);
      caffe_cpu_scale(d, Dtype(pow(a, -0.5)), bottom_diff+i*d, bottom_diff+i*d);
     }
	}

#ifdef CPU_ONLY
	STUB_GPU(L2NormalizationLayer);
#endif

	INSTANTIATE_CLASS(L2NormalizationLayer);
	REGISTER_LAYER_CLASS(L2Normalization);

}  // namespace caffe
