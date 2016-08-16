#ifndef CAFFE_RESHAPE_LAYER_HPP_
#define CAFFE_RESHAPE_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/*
 * @brief Reshapes the input Blob into an arbitrary-sized output Blob.
 */
template <typename Dtype>
class ReshapeLayer : public Layer<Dtype> {
 public:
  explicit ReshapeLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top);

 protected:
  virtual Dtype Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) { return Dtype(0.); }
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom) {}
  virtual Dtype Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) { return Dtype(0.); }
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom) {}
};

}  // namespace caffe

#endif  // CAFFE_RESHAPE_LAYER_HPP_
