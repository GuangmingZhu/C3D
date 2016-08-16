#!/usr/bin/env sh

NOWTIME=`date '+%Y%m%d-%H%M%S.%5N'`
GLOG_logtostderr=1 ../../build/tools/finetune_net.bin \
  chalearn_isogr_depth_solver.prototxt c3d_ucf101_finetune_whole_iter_20000\
  2>&1 | tee ./log/chalearn_isogr_depth_train.log.$NOWTIME
