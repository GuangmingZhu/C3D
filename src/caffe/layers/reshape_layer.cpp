#include <vector>

#include "caffe/reshape_layer.hpp"

namespace caffe {

template <typename Dtype>
void ReshapeLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  CHECK_EQ(bottom.size(), 1) << "Reshape Layer takes a single blob as input.";
  CHECK_EQ(top->size(), 1) << "Reshape Layer takes a single blob as output.";
  CHECK_EQ(this->layer_param_.reshape_param().shape_size(), 5) 
  	<< "Reshape Layer takes 5-dimensional reshape.";
  vector<int> top_shape(5, 0);
  int dim_to_infer = -1;
  for (int i = 0; i < 5; i++) {
	  top_shape[i] = this->layer_param_.reshape_param().shape(i);
	  if (top_shape[i] == -1) {
		  if (dim_to_infer == -1) {
			  dim_to_infer = i;
		  } else {
			  CHECK_EQ(dim_to_infer, -1) << "Only one dimensionty can be inferred";
		  }
	  }
  }
  if (dim_to_infer == -1) {
	  int top_count = 1;
	  for (int i = 0; i < 5; i++) {
		  top_count *= top_shape[i];
	  }
	  CHECK_EQ(top_count, bottom[0]->count()) << "output count must match input count";
  } else {
	  int infer_shape = bottom[0]->count();
	  for (int i = 0; i < 5; i++) {
		  if (dim_to_infer != i) {
			  infer_shape /= top_shape[i];
		  }
	  }
	  top_shape[dim_to_infer] = infer_shape;
  }

  (*top)[0]->Reshape(top_shape[0], top_shape[1], top_shape[2], top_shape[3], top_shape[4]);

  (*top)[0]->ShareData(*bottom[0]);
  (*top)[0]->ShareDiff(*bottom[0]);
}

INSTANTIATE_CLASS(ReshapeLayer);

}  // namespace caffe
