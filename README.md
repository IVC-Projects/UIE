# UIE
Underwater Image Enhancement


#1. 
Test images with reference images: test_images_pair_XXXX.
The output images will be stored in the same folder, and named “xx_out.png”.
To test these images, you need to move these output images to an another folder, and run the python file "Measure_PSNR_SSIM_UIQM/measure_test.py", the mean value of PSNR、SSIM and UIQM will be calculated.


#2. 
To test images, you need to check the checkpoint_dir in main_test_drcan.py and the filenames and data_dir in main_test_drcan.py. Also, the network is need to be noticed, and check if the network is the same as the network when training the model

#3.
You can download the model from the link: https://drive.google.com/drive/folders/1r5k5YHVsUNUmX9cbZqCAzclCOLny8bT2?usp=sharing


