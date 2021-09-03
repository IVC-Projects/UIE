import tensorflow as tf
from tensorflow.python import pywrap_tensorflow
import os
import numpy as np
# model_dir = r"F:/chenlong/UIE_MRJL_ChinaMM2021/checkpoint_FINAL/coarse_256/"
# model_dir = r"F:/chenlong/UWGAN_UIE/checkpoint/checkpoints_l1_uieb/"
# F:\chenlong\UWCNN\checkpoint\coarse_230
# F:\chenlong\WaterNet\checkpoint_ori\coarse_112
#F:\chenlong\UIE_MRJL_ChinaMM2021\checkpoint_euvp_0819
#F:\chenlong_8\UnderWater\UIE_MRJL\the_model_lzk_had_tried\20210619-1-lzk_checkpoint_raw_ca\coarse_256
sess = tf.Session()
# 本来我们需要重新像上一个文件那样重新构建整个graph，但是利用下面这个语句就可以加载整个graph了，方便
new_saver = tf.train.import_meta_graph(r'F:\chenlong\UIE_MRJL_ChinaMM2021\checkpoint_FINAL\coarse_256\coarse.model-10000.meta')
new_saver.restore(sess, r'F:\chenlong\UIE_MRJL_ChinaMM2021\checkpoint_FINAL\coarse_256\coarse.model-10000')  # 加载模型中各种变量的值，注意这里不用文件的后缀


graph = tf.get_default_graph()

flops = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.float_operation())
print('*****GFLOPs: {}',(flops.total_float_ops/1000000000.0))


#MFLOPS
#UWCNN 393.109813077     1
#UWGAN  1371.745448551   8
#WaterNet 1937.278466427 16
#Ours 4370.732374505     8
#batch=1:
# UWCNN=393
# UWGAN=171
# WaterNet=121
# ours=546

#
# import tensorflow as tf
# import keras.backend as K
# from keras.applications.mobilenet import MobileNet
#
# run_meta = tf.RunMetadata()
# with tf.Session(graph=tf.Graph()) as sess:
#     K.set_session(sess)
#     net = MobileNet(alpha=.75, input_tensor=tf.placeholder('float32', shape=(1,32,32,3)))
#
#     opts = tf.profiler.ProfileOptionBuilder.float_operation()
#     flops = tf.profiler.profile(sess.graph, run_meta=run_meta, cmd='op', options=opts)
#
#     opts = tf.profiler.ProfileOptionBuilder.trainable_variables_parameter()
#     params = tf.profiler.profile(sess.graph, run_meta=run_meta, cmd='op', options=opts)
#
#     print("{:,} --- {:,}".format(flops.total_float_ops/1e9, params.total_parameters))
import tensorflow as tf
def stats_graph(graph):
    flops = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.float_operation())
    params = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.trainable_variables_parameter())
    print('*****GFLOPs: {};'.format(flops.total_float_ops/1e9))

stats_graph(graph)