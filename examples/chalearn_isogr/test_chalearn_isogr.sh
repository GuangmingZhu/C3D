#!/usr/bin/env sh

# Step 1. Extract features from RGB-Based Conv3DNet 
NOWTIME=`date '+%Y%m%d-%H%M%S.%5N'`
GLOG_logtostderr=1 ../../build/tools/extract_features.bin \
  chalearn_isogr_rgb_iter_45000 chalearn_isogr_rgb_test.prototxt \
  prob chalearn_isogr_rgb_test_rst \
  784 GPU 0\
  2>&1 | tee ./log/chalearn_isogr_rgb_test.log.$NOWTIME

# Step 2. Extract features from Depth-Based Conv3DNet 
NOWTIME=`date '+%Y%m%d-%H%M%S.%5N'`
GLOG_logtostderr=1 ../../build/tools/extract_features.bin \
  chalearn_isogr_depth_iter_45000 chalearn_isogr_depth_test.prototxt \
  prob chalearn_isogr_depth_test_rst \
  784 GPU 0\
  2>&1 | tee ./log/chalearn_isogr_depth_test.log.$NOWTIME

# Step 3. Fuse the testing results in Steps 1 and 2 
python isogr_parse_result.py > test_prediction.txt 
