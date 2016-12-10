
""" Siamese implementation using Tensorflow with MNIST example.
This siamese network embeds a 28x28 image (a point in 784D) 
into a point in 2D.

By Youngwook Paul Kwon (young at berkeley.edu)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

#import system things
from tensorflow.examples.tutorials.mnist import input_data # for data
import tensorflow as tf
import tensorlayer as tl
import numpy as np
import os

import sys

#import helpers
import inference
import visualize

import code

#get this current directory
this_current_directory = os.getcwd()

# prepare data and tf.session
mnist = input_data.read_data_sets('MNIST_data', one_hot=False)
sess = tf.InteractiveSession()

# setup siamese network
batch_size = 256
siamese = inference.siamese_tl3(batch_size);
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(siamese.loss)
saver = tf.train.Saver()
tf.global_variables_initializer().run()

# for tensorboard
#summary_writer = tf.train.SummaryWriter(this_current_directory, graph_def=sess.graph_def)

# if you just want to load a previously trainmodel?
new = True
model_ckpt = os.path.join(this_current_directory,'model.ckpt.meta')
if os.path.isfile(model_ckpt):
    input_var = None
    while input_var not in ['yes', 'no']:
        if sys.version_info[0] >= 3:
            input_var = input("We found model.ckpt file. Do you want to load it [yes/no]?")
        else:
            input_var = raw_input("We found model.ckpt file. Do you want to load it [yes/no]?")
    if input_var == 'yes':
        new = False

# start training
if new:
    for step in range(100000):
        batch_x1, batch_y1 = mnist.train.next_batch(batch_size)
        batch_x2, batch_y2 = mnist.train.next_batch(batch_size)
        batch_y = (batch_y1 == batch_y2).astype('float')

        _, loss_v = sess.run([train_step, siamese.loss], feed_dict={
                            siamese.x: np.vstack((batch_x1,batch_x2)),  
                            siamese.y_: batch_y})

        if np.isnan(loss_v):
            print('Model diverged with loss = NaN')
            quit()

        if step % 10 == 0:
            print ('step %d: loss %.3f' % (step, loss_v))

        if step % 1000 == 0 and step > 0:
            saver.save(sess, os.path.join(this_current_directory,'model.ckpt'))
            embed = siamese.predict(mnist.test.images, sess)
            embed.tofile(os.path.join(this_current_directory,'embed.txt'))
else:
    saver.restore(sess, os.path.join(this_current_directory,'model.ckpt'))

    embed = np.fromfile(os.path.join(this_current_directory,'embed.txt'), dtype=np.float32)
    embed = embed.reshape([-1, 2])

#code.interact(local=dict(globals(),**locals()))
# visualize result
x_test = mnist.test.images.reshape([-1, 28, 28])
visualize.visualize(embed, x_test)
