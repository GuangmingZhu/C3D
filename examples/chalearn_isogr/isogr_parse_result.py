#!/usr/bin/env python

import sys
sys.path.append('../../python')
import caffe
import io
from PIL import Image
import numpy as np
import math
import leveldb

video_list = 'isogr_images_split/test_list.txt'
f = open(video_list, 'r')
f_lines = f.readlines()
f.close()

linenum = 0
predict_right = 0;
db1 = leveldb.LevelDB('./chalearn_isogr_rgb_test_rst')
db2 = leveldb.LevelDB('./chalearn_isogr_depth_test_rst')
datum1 = caffe.proto.caffe_pb2.Datum()
datum2 = caffe.proto.caffe_pb2.Datum()
for idx, line in enumerate(f_lines):
  key_str = '%06d' %(linenum)
  value1 = db1.Get(key_str)
  value2 = db2.Get(key_str)
  datum1.ParseFromString(value1)
  datum2.ParseFromString(value2)
  data1 = caffe.io.datum_to_array(datum1)
  data2 = caffe.io.datum_to_array(datum2)
  data  = data1*0.5 + data2*0.5
  #gt_label = int(line.split(' ')[2])
  #pd_label = np.argmax(data)+1
  #if gt_label == pd_label:
  #  predict_right = predict_right + 1  
  print '%s %s' %(line.strip('\n'), np.argmax(data)+1)
  linenum = linenum + 1

#print 'Accuracy = %f' %(float(predict_right)/linenum)  
