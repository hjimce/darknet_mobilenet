#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"

extern "C" {
#include "depthwise_convolutional_layer.h"
#include "batchnorm_layer.h"
#include "gemm.h"
#include "blas.h"
#include "im2col.h"
#include "col2im.h"
#include "utils.h"
#include "cuda.h"
}

//修改，未调试
void forward_depthwise_convolutional_layer_gpu(depthwise_convolutional_layer l, network net)
{
	//cuda_pull_array(l.output_gpu, l.output, l.c*l.out_h*l.out_w);//add by hjimce for debug
    fill_gpu(l.outputs*l.batch, 0, l.output_gpu, 1);

/*#ifdef CUDNN
    float one = 1;
    cudnnConvolutionForward(cudnn_handle(),
                &one,
                l.srcTensorDesc,
                net.input_gpu,
                l.weightDesc,
                l.weights_gpu,
                l.convDesc,
                l.fw_algo,
                net.workspace,
                l.workspace_size,
                &one,
                l.dstTensorDesc,
                l.output_gpu);

#else*/
    int i;
    int k = l.size*l.size;
    int n = l.out_w*l.out_h;

	for (int b = 0; b < l.batch; ++b) {
		for (int c = 0; c<l.c; c++)
		{
			float *aoffset = l.weights_gpu + c*l.size*l.size;
			float *boffset = net.workspace;
			float *coffset = l.output_gpu + c*l.out_h*l.out_w + b*l.n*l.out_h*l.out_w;
			float *intput_offset = net.input_gpu + c*l.h*l.w + b*l.c*l.h*l.w;
			im2col_gpu(intput_offset, 1, l.h, l.w,
				l.size, l.stride, l.pad, boffset);
			gemm_gpu(0, 0, 1, n, k, 1, aoffset, k, boffset, n, 1, coffset, n);

		}
	}

//#endif

    if (l.batch_normalize) {
        forward_batchnorm_layer_gpu(l, net);
    } else {
        add_bias_gpu(l.output_gpu, l.biases_gpu, l.batch, l.n, l.out_w*l.out_h);
    }
	int m = l.n;
    activate_array_gpu(l.output_gpu, l.outputs*l.batch, l.activation);


	//cuda_pull_array(l.output_gpu, l.output, l.c*l.out_h*l.out_w);//add by hjimce for debug

	

}

//修改，未调试
void backward_depthwise_convolutional_layer_gpu(depthwise_convolutional_layer l, network net)
{

    constrain_gpu(l.outputs*l.batch, 1, l.delta_gpu, 1);
    gradient_array_gpu(l.output_gpu, l.outputs*l.batch, l.activation, l.delta_gpu);


    if(l.batch_normalize){
        backward_batchnorm_layer_gpu(l, net);
    } else {
        backward_bias_gpu(l.bias_updates_gpu, l.delta_gpu, l.batch, l.n, l.out_w*l.out_h);
    }
    float *original_input = net.input_gpu;


/*#ifdef CUDNN
    float one = 1;
    cudnnConvolutionBackwardFilter(cudnn_handle(),
            &one,
            l.srcTensorDesc,
            net.input_gpu,
            l.ddstTensorDesc,
            l.delta_gpu,
            l.convDesc,
            l.bf_algo,
            net.workspace,
            l.workspace_size,
            &one,
            l.dweightDesc,
            l.weight_updates_gpu);

    if(net.delta_gpu){
        cudnnConvolutionBackwardData(cudnn_handle(),
                &one,
                l.weightDesc,
                l.weights_gpu,
                l.ddstTensorDesc,
                l.delta_gpu,
                l.convDesc,
                l.bd_algo,
                net.workspace,
                l.workspace_size,
                &one,
                l.dsrcTensorDesc,
                net.delta_gpu);

    }

#else*/
    int m = l.n;
    int n = l.size*l.size;
    int k = l.out_w*l.out_h;
	//pull_depthwise_convolutional_layer(l);//add by hjimce for debug
	for (int b = 0; b < l.batch; ++b) {
		for (int c = 0; c<l.c; c++)
		{


			//对权重求导
			float *aoffset = l.delta_gpu + c*l.out_h*l.out_w + b*l.n*l.out_h*l.out_w;
			float *boffset = net.workspace;
			float *coffset = l.weight_updates_gpu + c*l.size*l.size;


			float *im = net.input_gpu + c*l.h*l.w + b*l.c*l.h*l.w;


			im2col_gpu(im, 1, l.h, l.w,
				l.size, l.stride, l.pad, boffset);
			gemm_gpu(0, 1, 1, n, k, 1, aoffset, k, boffset, k, 1, coffset, n);
			//对本层网络输入求导：也就是用原始的权重，对输出层数据微分特征图进行卷积运算

			if (net.delta_gpu) {
				aoffset = l.weights_gpu + c*l.size*l.size;
				boffset = l.delta_gpu + c*l.out_h*l.out_w + b*l.n*l.out_h*l.out_w;
				coffset = net.workspace;

				gemm_gpu(1, 0, n, k, 1, 1, aoffset, n, boffset, k, 0, coffset, k);

				col2im_gpu(net.workspace, 1, l.h, l.w, l.size, l.stride, l.pad, net.delta_gpu + c*l.h*l.w + b*l.n*l.h*l.w);
			}


		}
	}

	//pull_depthwise_convolutional_layer(l);//测试求导结果add by hjimce for debug

//#endif
}
//修改，未调试
void pull_depthwise_convolutional_layer(depthwise_convolutional_layer layer)
{
    cuda_pull_array(layer.weights_gpu, layer.weights, layer.c*layer.size*layer.size);
    cuda_pull_array(layer.biases_gpu, layer.biases, layer.n);
    cuda_pull_array(layer.weight_updates_gpu, layer.weight_updates, layer.c*layer.size*layer.size);
    cuda_pull_array(layer.bias_updates_gpu, layer.bias_updates, layer.n);
    if (layer.batch_normalize){
        cuda_pull_array(layer.scales_gpu, layer.scales, layer.n);
        cuda_pull_array(layer.rolling_mean_gpu, layer.rolling_mean, layer.n);
        cuda_pull_array(layer.rolling_variance_gpu, layer.rolling_variance, layer.n);
    }
}
//修改，未调试
void push_depthwise_convolutional_layer(depthwise_convolutional_layer layer)
{
    cuda_push_array(layer.weights_gpu, layer.weights, layer.c*layer.size*layer.size);
    cuda_push_array(layer.biases_gpu, layer.biases, layer.n);
    cuda_push_array(layer.weight_updates_gpu, layer.weight_updates, layer.c*layer.size*layer.size);
    cuda_push_array(layer.bias_updates_gpu, layer.bias_updates, layer.n);
    if (layer.batch_normalize){
        cuda_push_array(layer.scales_gpu, layer.scales, layer.n);
        cuda_push_array(layer.rolling_mean_gpu, layer.rolling_mean, layer.n);
        cuda_push_array(layer.rolling_variance_gpu, layer.rolling_variance, layer.n);
    }
}
//修改，未调试
void update_depthwise_convolutional_layer_gpu(layer l, update_args a)
{
    float learning_rate = a.learning_rate*l.learning_rate_scale;
    float momentum = a.momentum;
    float decay = a.decay;
    int batch = a.batch;

    int size = l.size*l.size*l.c;

    if(a.adam){
        adam_update_gpu(l.weights_gpu, l.weight_updates_gpu, l.m_gpu, l.v_gpu, a.B1, a.B2, a.eps, decay, learning_rate, size, batch, a.t);
        adam_update_gpu(l.biases_gpu, l.bias_updates_gpu, l.bias_m_gpu, l.bias_v_gpu, a.B1, a.B2, a.eps, decay, learning_rate, l.n, batch, a.t);
        if(l.scales_gpu){
            adam_update_gpu(l.scales_gpu, l.scale_updates_gpu, l.scale_m_gpu, l.scale_v_gpu, a.B1, a.B2, a.eps, decay, learning_rate, l.n, batch, a.t);
        }
    }else{
        axpy_gpu(size, -decay*batch, l.weights_gpu, 1, l.weight_updates_gpu, 1);
        axpy_gpu(size, learning_rate/batch, l.weight_updates_gpu, 1, l.weights_gpu, 1);
        scal_gpu(size, momentum, l.weight_updates_gpu, 1);

        axpy_gpu(l.n, learning_rate/batch, l.bias_updates_gpu, 1, l.biases_gpu, 1);
        scal_gpu(l.n, momentum, l.bias_updates_gpu, 1);

        if(l.scales_gpu){
            axpy_gpu(l.n, learning_rate/batch, l.scale_updates_gpu, 1, l.scales_gpu, 1);
            scal_gpu(l.n, momentum, l.scale_updates_gpu, 1);
        }
    }
}


