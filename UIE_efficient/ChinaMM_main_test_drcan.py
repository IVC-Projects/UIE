import time

from ChinaMM_model_drcan import T_CNN
from utils import *
import numpy as np
import tensorflow as tf

import pprint
import os

flags = tf.app.flags
flags.DEFINE_integer("epoch", 15, "Number of epoch [120]")
flags.DEFINE_integer("batch_size", 16, "The size of batch images [128]")
flags.DEFINE_integer("image_height", 256, "The size of image to use [230]")
flags.DEFINE_integer("image_width", 256, "The size of image to use [310]")
flags.DEFINE_integer("label_height", 256, "The size of label to produce [230]")
flags.DEFINE_integer("label_width", 256, "The size of label to produce [310]")
flags.DEFINE_float("learning_rate", 0.001, "The learning rate of gradient descent algorithm [1e-4]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_integer("c_dim", 3, "Dimension of image color. [3]")
flags.DEFINE_integer("c_depth_dim", 1, "Dimension of depth. [1]")
flags.DEFINE_string("checkpoint_dir", "checkpoint_FINAL0", "Name of checkpoint directory [checkpoint]")
flags.DEFINE_string("sample_dir", "sample", "Name of sample directory [sample]")
flags.DEFINE_string("test_data_dir", "train_and_test_dataset_euvp_uwscences/input_test", "Name of sample directory [test]")
flags.DEFINE_boolean("is_train", False, "True for training, False for testing [True]")
FLAGS = flags.FLAGS

pp = pprint.PrettyPrinter()

def main(_):
  pp.pprint(flags.FLAGS.__flags)


  if not os.path.exists(FLAGS.checkpoint_dir):
    os.makedirs(FLAGS.checkpoint_dir)
  if not os.path.exists(FLAGS.sample_dir):
    os.makedirs(FLAGS.sample_dir)
  filenames = os.listdir('test_images_pair')#input_test
  data_dir = os.path.join(os.getcwd(), 'test_images_pair')
  data = glob.glob(os.path.join(data_dir, "*.png"))
  test_data_list = data + glob.glob(os.path.join(data_dir, "*.jpg"))+glob.glob(os.path.join(data_dir, "*.bmp"))+glob.glob(os.path.join(data_dir, "*.png"))


  # filenames5 = os.listdir('./train_and_test_dataset_uieb/input_sm_test')
  # data_dir5 = os.path.join(os.getcwd(), './train_and_test_dataset_uieb/input_sm_test')
  # data5 = glob.glob(os.path.join(data_dir5, "*.png"))
  # test_data_list5 = data5 + glob.glob(os.path.join(data_dir5, "*.jpg")) + glob.glob(os.path.join(data_dir5, "*.bmp")) + glob.glob(os.path.join(data_dir5, "*.jpeg"))

  times = []; s = time.time()
  for ide in range(0,len(test_data_list)):
    image_test =  get_image(test_data_list[ide],is_grayscale=False)

    #sm_test = get_image(test_data_list5[ide], is_grayscale=False)
    shape = image_test.shape
    tf.reset_default_graph()
    with tf.Session() as sess:
      # with tf.device('/cpu:0'):
        s = time.time()
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
                  #test_sm_name=test_data_list5[ide],
                  id = ide
                  )
        tot = time.time() - s
        times.append(tot)
        srcnn.train(FLAGS)
        sess.close()
    tf.get_default_graph().finalize()
    Ttime, Mtime = np.sum(times[1:]), np.mean(times[1:])
    print("Time taken: {0} sec at {1} fps".format(Ttime, 1. / Mtime))
if __name__ == '__main__':
  tf.app.run()
