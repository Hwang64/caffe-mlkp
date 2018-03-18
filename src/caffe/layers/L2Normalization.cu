#include <vector>

#include "caffe/util/math_functions.hpp"
#include "caffe/layers/L2Norm.hpp"

namespace caffe {

	template <typename Dtype>
	__global__ void L2NormalizationFillScale_across(const int nthreads, const Dtype* const in, //��������ƽ����
		const int num, const int channels, const int height,
		const int width, const int size, Dtype* const scale) {
		CUDA_KERNEL_LOOP(index, nthreads) {
			// find out the local offset
			const int w = index % width;
			const int h = (index / width) % height;
			const int n = index / width / height;
			const int offset = (n * channels * height + h) * width + w;
			const int step = height * width;
			const Dtype* const in_off = in + offset;
			Dtype* const scale_off = scale + offset;
			int head = 0;
			Dtype accum_scale = 0;
			// fill the scale at [n, :, h, w]
			// accumulate values ƽ���Ӻ�
			while (head < channels) {
				accum_scale += in_off[head * step] * in_off[head * step];
				++head;
			}
			head = 0;
			while (head < channels){
				scale_off[head * step] = 1e-10 + accum_scale;
				++head;
			}
		}
	}

template <typename Dtype>
__global__ void L2NormalizationFillScale_all(const int nthreads, const Dtype* const in, //��������ƽ����
	const int num, const int channels, const int height,
	const int width, const int size, Dtype* const scale) {
	CUDA_KERNEL_LOOP(index, nthreads) {
		// find out the local offset
		const int w = index % width;
		const int h = (index / width) % height;
		const int c = (index / channels / height) % channels;
		const int n = index / width / height / channels;
		const int offset = ((n * channels + c) * height + h) * width + w;
		const int step = 1;
		const Dtype* const in_off = in + offset;
		Dtype* const scale_off = scale + offset;
		int head = 0;
		Dtype accum_scale = 0;
		// fill the scale at [n, :, h, w]
		// accumulate values ƽ���Ӻ�
		while (head < channels * height * width) {
			accum_scale += in_off[head * step] * in_off[head * step];
			++head;
		}
		head = 0;
		while (head < channels * height * width){
			scale_off[head * step] = 1e-10 + accum_scale;
			++head;
		}
	}
}


	template <typename Dtype>
	void L2NormalizationLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		switch (this->layer_param_.l2n_param().norm_region()) {
		case L2Parameter_NormRegion_ACROSS_CHANNELS:
			ChannelForward_gpu(bottom, top);
			break;
		case L2Parameter_NormRegion_ALL_ENTRY:
			AllForward_gpu(bottom, top);
			break;
		default:
			LOG(FATAL) << "Unknown normalization region.";
		}
	}

	// TODO: check if it would be faster to just put it into the previous kernel.
	template <typename Dtype>
	__global__ void L2NormalizationComputeOutput(const int nthreads, const Dtype* const in,//�������������Ŀ�����ͬʱԭ��Ԫ��
																							//�������������͵�
		const Dtype* const scale, const Dtype negative_beta, Dtype* const out) {
		CUDA_KERNEL_LOOP(index, nthreads) {
			out[index] = in[index] * pow(scale[index], negative_beta);
		}
	}

	template <typename Dtype>
	void L2NormalizationLayer<Dtype>::ChannelForward_gpu(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
		// First, compute scale
		const Dtype* bottom_data = bottom[0]->gpu_data();
		Dtype* top_data = top[0]->mutable_gpu_data();
		Dtype* scale_data = scale_.mutable_gpu_data();
		// We will launch one kernel for each pixel location, and have the kernel
		// go through all the channels.
		int n_threads = num_ * height_ * width_;
		// NOLINT_NEXT_LINE(whitespace/operators)
		L2NormalizationFillScale_across << <CAFFE_GET_BLOCKS(n_threads), CAFFE_CUDA_NUM_THREADS >> >(
			n_threads, bottom_data, num_, channels_, height_, width_, size_, scale_data);
		CUDA_POST_KERNEL_CHECK;
		n_threads = bottom[0]->count();
		// NOLINT_NEXT_LINE(whitespace/operators)
		L2NormalizationComputeOutput << <CAFFE_GET_BLOCKS(n_threads), CAFFE_CUDA_NUM_THREADS >> >(
			n_threads, bottom_data, scale_data, -beta_, top_data);
		CUDA_POST_KERNEL_CHECK;
	}

	// version 2.0
	template <typename Dtype>
	void L2NormalizationLayer<Dtype>::AllForward_gpu(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
		// First, compute scale
		const Dtype* bottom_data = bottom[0]->gpu_data();
		Dtype* top_data = top[0]->mutable_gpu_data();
		Dtype* scale_data = scale_.mutable_gpu_data();
		// We will launch one kernel for each pixel location, and have the kernel
		// go through all the channels.
		int n_threads = num_;
		// NOLINT_NEXT_LINE(whitespace/operators)
		L2NormalizationFillScale_all << <CAFFE_GET_BLOCKS(n_threads), CAFFE_CUDA_NUM_THREADS >> >(
			n_threads, bottom_data, num_, channels_, height_, width_, size_, scale_data);
		CUDA_POST_KERNEL_CHECK;
		n_threads = bottom[0]->count();
		// NOLINT_NEXT_LINE(whitespace/operators)
		L2NormalizationComputeOutput << <CAFFE_GET_BLOCKS(n_threads), CAFFE_CUDA_NUM_THREADS >> >(
			n_threads, bottom_data, scale_data, -beta_, top_data);
		CUDA_POST_KERNEL_CHECK;
	}


	// version 1.0
  /*
	template <typename Dtype>
  void L2NormalizationLayer<Dtype>::AllForward_gpu(
		const vector<Blob<Dtype>*>& bottom,const vector<Blob<Dtype>*>& top){
			const Dtype* bottom_data = bottom[0]->gpu_data();
			Dtype* top_data = top[0]->mutable_gpu_data();
			//Dtype* scale_data = scale_.mutable_cpu_data();
			Blob<Dtype> square(1, channels_, height_, width_);
  		Dtype* square_data = square.mutable_gpu_data();
			for (int n = 0; n < num_; ++n) {
  			// compute the padded square
				caffe_gpu_set(square.count(), Dtype(0), square_data);
  			caffe_gpu_powx(channels_ * height_ * width_,
				 bottom_data + bottom[0]->offset(n),Dtype(2.0),square_data);
	  		 Dtype normsqr;
				 caffe_gpu_asum<Dtype>(channels_ * height_ * width_,
				 square_data,&normsqr);
	  	   caffe_gpu_scale<Dtype>(channels_ * height_ * width_,pow(normsqr+1e-10,-beta_),
				 bottom_data+bottom[0]->offset(n),top_data+top[0]->offset(n));
	  	}
	}
	*/
	template void L2NormalizationLayer<float>::ChannelForward_gpu(
	    const vector<Blob<float>*>& bottom, const vector<Blob<float>*>& top);
	template void L2NormalizationLayer<double>::ChannelForward_gpu(
			const vector<Blob<double>*>& bottom, const vector<Blob<double>*>& top);
	template void L2NormalizationLayer<float>::AllForward_gpu(
			const vector<Blob<float>*>& bottom, const vector<Blob<float>*>& top);
	template void L2NormalizationLayer<double>::AllForward_gpu(
	    const vector<Blob<double>*>& bottom, const vector<Blob<double>*>& top);

	template <typename Dtype>
	void L2NormalizationLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
		switch (this->layer_param_.lrn_param().norm_region()) {
		case L2Parameter_NormRegion_ACROSS_CHANNELS:
			ChannelBackward_gpu(top, propagate_down, bottom);
			break;
		case L2Parameter_NormRegion_ALL_ENTRY:
			AllBackward_gpu(top, propagate_down, bottom);
			break;
		default:
			LOG(FATAL) << "Unknown normalization region.";
		}
	}

	template <typename Dtype>
	__global__ void L2NormalizationComputeDiff_across(const int nthreads,
		const Dtype* const bottom_data, const Dtype* const top_data,
		const Dtype* const scale, const Dtype* const top_diff,
		const int num, const int channels, const int height,
		const int width, const int size, const Dtype negative_beta,
		const Dtype cache_ratio, Dtype* const bottom_diff) {
		CUDA_KERNEL_LOOP(index, nthreads) {
			// find out the local offset
			const int w = index % width;
			const int h = (index / width) % height;
			const int n = index / width / height;
			const int offset = (n * channels * height + h) * width + w;
			const int step = height * width;
			//const Dtype* const bottom_off = bottom_data + offset;
			const Dtype* const top_off = top_data + offset;
			const Dtype* const scale_off = scale + offset;
			const Dtype* const top_diff_off = top_diff + offset;
			Dtype* const bottom_diff_off = bottom_diff + offset;
			int head = 0;
			Dtype accum_ratio = 0;
			// accumulate values
			while (head < channels) {
				accum_ratio += top_diff_off[head * step] * top_off[head * step];
				++head;
			}
			head = 0;
			while (head < channels){
				bottom_diff_off[head * step] = \
					top_diff_off[head * step]\
					* pow(scale_off[head * step], negative_beta)\
					- cache_ratio * top_off[head  * step] * accum_ratio * pow(scale_off[head*step], negative_beta);
				++head;
			}
		}
	}


	template <typename Dtype>
	__global__ void L2NormalizationComputeDiff_all(const int nthreads,
		const Dtype* const bottom_data, const Dtype* const top_data,
		const Dtype* const scale, const Dtype* const top_diff,
		const int num, const int channels, const int height,
		const int width, const int size, const Dtype negative_beta,
		const Dtype cache_ratio, Dtype* const bottom_diff) {
		CUDA_KERNEL_LOOP(index, nthreads) {
			// find out the local offset
			const int w = index % width;
			const int h = (index / width) % height;
			const int c = (index / width / height) % channels;
			const int n = index / width / height / channels;
			const int offset = (( n * channels + c ) * height + h) * width + w;
			const int step = 1;
			//const Dtype* const bottom_off = bottom_data + offset;
			const Dtype* const top_off = top_data + offset;
			const Dtype* const scale_off = scale + offset;
			const Dtype* const top_diff_off = top_diff + offset;
			Dtype* const bottom_diff_off = bottom_diff + offset;
			int head = 0;
			Dtype accum_ratio = 0;
			// accumulate values
			while (head < channels * height * width) {
				accum_ratio += top_diff_off[head * step] * top_off[head * step];
				++head;
			}
			head = 0;
			while (head < channels * height * width){
				bottom_diff_off[head * step] = \
					top_diff_off[head * step]\
					* pow(scale_off[head * step], negative_beta)\
					- cache_ratio * top_off[head  * step] * accum_ratio * pow(scale_off[head*step], negative_beta);
				++head;
			}
		}
	}


	template <typename Dtype>
	void L2NormalizationLayer<Dtype>::ChannelBackward_gpu(
		const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
		const vector<Blob<Dtype>*>& bottom) {
		int n_threads = num_ * height_ * width_;
		// NOLINT_NEXT_LINE(whitespace/operators)
		L2NormalizationComputeDiff_across << <CAFFE_GET_BLOCKS(n_threads), CAFFE_CUDA_NUM_THREADS >> >(
			n_threads, bottom[0]->gpu_data(), top[0]->gpu_data(),
			scale_.gpu_data(), top[0]->gpu_diff(), num_, channels_, height_, width_,
			size_, -beta_, Dtype(1.),
			bottom[0]->mutable_gpu_diff());
	}
	// version 2.0

	template <typename Dtype>
	void L2NormalizationLayer<Dtype>::AllBackward_gpu(
		const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
		const vector<Blob<Dtype>*>& bottom) {
		int n_threads = num_;
		// NOLINT_NEXT_LINE(whitespace/operators)
		L2NormalizationComputeDiff_all << <CAFFE_GET_BLOCKS(n_threads), CAFFE_CUDA_NUM_THREADS >> >(
			n_threads, bottom[0]->gpu_data(), top[0]->gpu_data(),
			scale_.gpu_data(), top[0]->gpu_diff(), num_, channels_, height_, width_,
			size_, -beta_, Dtype(1.),
			bottom[0]->mutable_gpu_diff());
	}

	// version 1.0
 /*
	template <typename Dtype>
	void L2NormalizationLayer<Dtype>::AllBackward_gpu(
		const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
		const vector<Blob<Dtype>*>& bottom){
			const Dtype* top_diff = top[0]->gpu_diff();
			const Dtype* top_data = top[0]->gpu_data();
			const Dtype* bottom_data = bottom[0]->gpu_data();
			Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
			int n = top[0]->num();
			int d = top[0]->count() / n;
			for (int i=0; i<n; ++i) {
			Dtype a;
			caffe_gpu_dot(d, top_data+i*d, top_diff+i*d,&a);
			caffe_gpu_scale(d, a, top_data+i*d, bottom_diff+i*d);
			caffe_gpu_sub(d, top_diff+i*d, bottom_diff+i*d, bottom_diff+i*d);
			caffe_gpu_dot(d, bottom_data+i*d, bottom_data+i*d,&a);
			caffe_gpu_scale(d, Dtype(pow(a+1e-10, -beta_)), bottom_diff+i*d, bottom_diff+i*d);
		}
	}
  */

	template void L2NormalizationLayer<float>::ChannelBackward_gpu(
	    const vector<Blob<float>*>& bottom,
			const vector<bool>& propagate_down,const vector<Blob<float>*>& top);
	template void L2NormalizationLayer<double>::ChannelBackward_gpu(
			const vector<Blob<double>*>& bottom,
			const vector<bool>& propagate_down,const vector<Blob<double>*>& top);
  template void L2NormalizationLayer<float>::AllBackward_gpu(
			const vector<Blob<float>*>& bottom,
			const vector<bool>& propagate_down,const vector<Blob<float>*>& top);
	template void L2NormalizationLayer<double>::AllBackward_gpu(
	    const vector<Blob<double>*>& bottom,
			const vector<bool>& propagate_down,const vector<Blob<double>*>& top);

	INSTANTIATE_LAYER_GPU_FUNCS(L2NormalizationLayer);

}  // namespace caffe
