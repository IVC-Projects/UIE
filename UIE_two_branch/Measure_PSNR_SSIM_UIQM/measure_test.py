"""
# > Script for measuring quantitative performances in terms of
#    - Underwater Image Quality Measure (UIQM)
#    - Structural Similarity Metric (SSIM)
#    - Peak Signal to Noise Ratio (PSNR)
#
#    You should check the folder path where the pictures are stored
#    Also, you should check if the suffix of the picture in the folder is "_out.png"
# > step:
#    - Put the raw underwater images into the folder REAL_im_dir
#    - Put the generation images into the folder GEN_im_dir
#    - Put the ground truth images into the folder GTr_im_dir
"""
## python libs
import os
import ntpath
import numpy as np
from scipy import misc
## local libs
from Measure_PSNR_SSIM_UIQM.data_utils import getPaths
from Measure_PSNR_SSIM_UIQM.uqim_utils import getUIQM
from Measure_PSNR_SSIM_UIQM.ssm_psnr_utils import getSSIM, getPSNR
## data paths
## UIEB-90
# REAL_im_dir = r'F:\chenlong_8\UnderWater\UIE_MRJL_Only_One_Input\test_images_pair/'  # raw underwater images
# GEN_im_dir  = r"F:\chenlong_8\UnderWater\UIE_MRJL_Only_One_Input\20210825_01_lzksnetwork_output_use_uieb_No2_Branch_uieb90/"  # generator images
# GTr_im_dir  = r'F:\chenlong_8\UnderWater\UIE_MRJL_Only_One_Input\test_groundtruth/'  # ground truth images

# ## UFO-120
# REAL_im_dir = r'F:\chenlong_8\UnderWater\UIE_MRJL_Only_One_Input\test_images_pair_ufo120/'  # raw underwater images
# GEN_im_dir  = r"F:\chenlong\UWGAN_UIE\test_images_pair_output_retrain_use_UFO_ufo120/"  # generator images
# GTr_im_dir  = r'F:\chenlong_8\UnderWater\UIE_MRJL_Only_One_Input\test_groundtruth_ufo120/'  # ground truth images

## EUVP-185
REAL_im_dir = r'F:\chenlong_8\UnderWater\UIE_MRJL_Only_One_Input\test_images_pair_euvp_uwscences_185/'  # raw underwater images
GEN_im_dir  = r"F:\chenlong_8\UnderWater\UIE_MRJL_Only_One_Input\20210825_01_lzksnetwork_output_use_uieb_No2_Branch_euvp185/"  # generator images
GTr_im_dir  = r'F:\chenlong_8\UnderWater\UIE_MRJL_Only_One_Input\test_groundtruth_euvp_uwscences_185/'  # ground truth images

## Synthetic 160
# REAL_im_dir = r'F:\chenlong_8\UnderWater\UIE_MRJL_Only_One_Input\test_images_pair_synthetic_160/'  # raw underwater images
# GEN_im_dir  = r"F:\chenlong\FUnIE-GAN-master\data\FUnIE_test_output_retrain_use_synthetic_synthetic160/"  # generator images
# GTr_im_dir  = r'F:\chenlong_8\UnderWater\UIE_MRJL_Only_One_Input\test_groundtruth_synthetic160/'  # ground truth images
REAL_paths, GEN_paths = getPaths(REAL_im_dir), getPaths(GEN_im_dir)


## mesures uqim for all images in a directory
def measure_UIQMs(dir_name):
    paths = getPaths(dir_name)
    uqims = []
    for img_path in paths:
        im = misc.imread(img_path)
        uqims.append(getUIQM(im))
    return np.array(uqims)

## compares avg ssim and psnr
def measure_SSIM_PSNRs(GT_dir, Gen_dir):
    """
      Assumes:
        * GT_dir contain ground-truths {filename.ext}
        * Gen_dir contain generated images {filename_gen.png}
        * Images are of same-size
    """
    GT_paths, Gen_paths = getPaths(GT_dir), getPaths(Gen_dir)
    ssims, psnrs = [], []
    for img_path in GT_paths:
        name_split = ntpath.basename(img_path).split('.')
        gen_path = os.path.join(Gen_dir, name_split[0] + '_out.png') #+name_split[1])  png or jpg
        if (gen_path in Gen_paths):
            r_im = misc.imread(img_path)
            g_im = misc.imread(gen_path)
            assert (r_im.shape==g_im.shape), "The images should be of same-size"
            ssim = getSSIM(r_im, g_im)
            psnr = getPSNR(r_im, g_im)
            #print ("{0}, {1}: {2}".format(img_path,gen_path, ssim))
            #print ("{0}, {1}: {2}".format(img_path,gen_path, psnr))
            ssims.append(ssim)
            psnrs.append(psnr)
    return np.array(ssims), np.array(psnrs)

### compute SSIM and PSNR
SSIM_measures, PSNR_measures = measure_SSIM_PSNRs(GTr_im_dir, GEN_im_dir)
print ("SSIM >> Mean: {0} std: {1}".format(np.mean(SSIM_measures), np.std(SSIM_measures)))
print ("PSNR >> Mean: {0} std: {1}".format(np.mean(PSNR_measures), np.std(PSNR_measures)))##np.std:计算标准差

### compute and compare UIQMs
# g_truth = measure_UIQMs(GTr_im_dir)
# print ("G. Truth UQIM  >> Mean: {0} std: {1}".format(np.mean(g_truth), np.std(g_truth)))
gen_uqims = measure_UIQMs(GEN_im_dir)
print ("Generated UQIM >> Mean: {0} std: {1}".format(np.mean(gen_uqims), np.std(gen_uqims)))
# real_uqims = measure_UIQMs(REAL_im_dir)
# print ("Inputs UQIM   >> Mean: {0} std: {1}".format(np.mean(real_uqims), np.std(real_uqims)))


