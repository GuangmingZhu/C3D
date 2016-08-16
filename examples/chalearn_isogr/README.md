# ChaLearn_Isogr

# 1. Data Preparation
This model needs image-style input, So you need to extract images from IsoGD_phase_1 and IsoGD_phase_2 by running extract_frames.sh before you train or test the model. Do modifiy the ROOTDIR before you run extract_frames.sh.

# 2. Train or Test file lists
When you have converted video files into image files, you need to change the root path in the files under the isogr_images_split dir.  

# 3. Train the model
The model will be trained based on the pretrained model c3d_ucf101_finetune_whole_iter_20000. You can run train_chalearn_isogr_rgb.sh and train_chalearn_isogr_depth.sh to train the model based on the RGB images and the depth images, respectively.

# 4. Test the model
You can run test_chalearn_isogr.sh to test the model. Please check that the trained net param files have been downloaded and located in the right position. You will get test_prediction.txt which store the final test result when testing. 
