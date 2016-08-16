# Isogr_Images_Split
Line Format in each file: Image_Path Frame_Cnt Label
  1. Each Image_Path is corresponding to one video file of  IsoGD_Phase_1 or IsoGD_Phase_2;
  2. Frame_Cnt is the total frame count of the video file, i.e., the count of images in Image_Path;
  3. Label is the gesture label which is in range 0~248.

train_depth_list.txt: 
  corresponding to IsoGD_Phase_1/train_list.txt
train_valid_depth_list.txt:
  corresponding to IsoGD_Phase_1/train_list.txt and IsoGD_Phase_1/valid_list.txt
train_rgb_list.txt          
  corresponding to IsoGD_Phase_1/train_list.txt
train_valid_rgb_list.txt  
  corresponding to IsoGD_Phase_1/train_list.txt and IsoGD_Phase_1/valid_list.txt
valid_depth_list.txt
  corresponding to IsoGD_Phase_1/valid_list.txt
valid_rgb_list.txt
  corresponding to IsoGD_Phase_1/valid_list.txt
test_depth_list.txt  
  corresponding to IsoGD_Phase_2/test_list.txt
test_rgb_list.txt     
  corresponding to IsoGD_Phase_2/test_list.txt
