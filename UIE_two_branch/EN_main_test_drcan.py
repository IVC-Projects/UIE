import datetime
import time

from EN_model_drcan import T_CNN
from utils import *
import numpy as np
import tensorflow as tf
tf.enable_eager_execution()
import pprint
import os

flags = tf.app.flags
flags.DEFINE_integer("epoch", 600, "Number of epoch [120]")
flags.DEFINE_integer("batch_size", 16, "The size of batch images [128]")
flags.DEFINE_integer("image_height", 256, "The size of image to use [230]")
flags.DEFINE_integer("image_width", 256, "The size of image to use [310]")
flags.DEFINE_integer("label_height", 256, "The size of label to produce [230]")
flags.DEFINE_integer("label_width", 256, "The size of label to produce [310]")
flags.DEFINE_float("learning_rate", 0.001, "The learning rate of gradient descent algorithm [1e-4]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_integer("c_dim", 3, "Dimension of image color. [3]")
flags.DEFINE_integer("c_depth_dim", 1, "Dimension of depth. [1]")
flags.DEFINE_string("checkpoint_dir", "20210712_02_lzksnetwork_CHECKPOINT_use_ufo120_1500", "Name of checkpoint directory [checkpoint]")
flags.DEFINE_string("sample_dir", "sample", "Name of sample directory [sample]")
flags.DEFINE_string("test_data_dir", "test_images_pair_U45", "Name of sample directory [test]")
flags.DEFINE_boolean("is_train", False, "True for training, False for testing [True]")
FLAGS = flags.FLAGS

pp = pprint.PrettyPrinter()

def main(_):
  # pp.pprint(flags.FLAGS.__flags)

  if not os.path.exists(FLAGS.checkpoint_dir):
    os.makedirs(FLAGS.checkpoint_dir)
  if not os.path.exists(FLAGS.sample_dir):
    os.makedirs(FLAGS.sample_dir)
  filenames = os.listdir('test_images_pair_ufo120')#input_test
  data_dir = os.path.join(os.getcwd(), 'test_images_pair_ufo120')
  data = glob.glob(os.path.join(data_dir, "*.png"))
  test_data_list = data + glob.glob(os.path.join(data_dir, "*.jpg"))+glob.glob(os.path.join(data_dir, "*.bmp"))+glob.glob(os.path.join(data_dir, "*.png"))

  # filenames5 = os.listdir('./train_and_test_dataset_uieb/input_sm_test')
  # data_dir5 = os.path.join(os.getcwd(), './train_and_test_dataset_uieb/input_sm_test')
  # data5 = glob.glob(os.path.join(data_dir5, "*.png"))
  # test_data_list5 = data5 + glob.glob(os.path.join(data_dir5, "*.jpg")) + glob.glob(os.path.join(data_dir5, "*.bmp")) + glob.glob(os.path.join(data_dir5, "*.jpeg"))

  # filenames6 = os.listdir('./train_and_test_dataset_uieb/input_wb_test')
  # data_dir6 = os.path.join(os.getcwd(), './train_and_test_dataset_uieb/input_wb_test')
  # data6 = glob.glob(os.path.join(data_dir6, "*.png"))
  # test_data_list6 = data6 + glob.glob(os.path.join(data_dir6, "*.jpg")) + glob.glob(
  #     os.path.join(data_dir6, "*.bmp")) + glob.glob(os.path.join(data_dir6, "*.jpeg"))


  starttime = datetime.datetime.now()
  # print("hhhh",len(test_data_list))
  for ide in range(0,len(test_data_list)):
    image_test =  get_image(test_data_list[ide],is_grayscale=False)

    # sm_test = get_image(test_data_list5[ide], is_grayscale=False)
    # wb_test = get_image(test_data_list6[ide], is_grayscale=False)
    shape = image_test.shape
    tf.reset_default_graph()
    with tf.Session() as sess:
      # with tf.device('/cpu:0'):
        srcnn = T_CNN(sess, 
                  image_height=shape[0],
                  image_width=shape[1],  
                  label_height=FLAGS.label_height, 
                  label_width=FLAGS.label_width, 
                  batch_size=FLAGS.batch_size,
                  c_dim=FLAGS.c_dim, 
                  c_depth_dim=FLAGS.c_depth_dim,
                  checkpoint_dir=FLAGS.checkpoint_dir,
                  sample_dir=FLAGS.sample_dir,
                  test_image_name = test_data_list[ide],
                  # test_sm_name=test_data_list5[ide],
                  # test_wb_name=test_data_list6[ide],
                  id = ide
                  )

        srcnn.train(FLAGS)
        sess.close()
    Endtime = datetime.datetime.now()
    print('per-Time:', (Endtime - starttime)/len(test_data_list))
    tf.get_default_graph().finalize()
if __name__ == '__main__':
  tf.app.run()
