"""Fully connected autoencoder in Python with tensorflow"""

import sys

sys.path.append('/home/radlr/anaconda3/envs/lVGPU2/lib/python3.6/site-packages')

#hint from https://stackoverflow.com/questions/19876079/opencv-cannot-find-module-cv2

# math etc.
from scipy import misc
import numpy as np

#plotting
import matplotlib as mpl
from matplotlib import pyplot as plt

#image functions, esp. resizing
import cv2
#directory functions
import os

import tensorflow as tf
import time
t = time.time()

# Setup parameters and hyperparameters
imgWidth = 48
imgHeight = 48
myChanels = 3
n_visible = imgWidth *imgHeight*3
n_visibleRGB = imgWidth *imgHeight*3
n_hidden = 4096 # 2048 # hidden units
lR = 1e-4
myIter = 3000
dispIt = 10# display every th iteration
dORate = 0.33 # dropout Rate

corruption_level = 0.0

# create node for input data and corruption mask
X = tf.placeholder("float",[None, n_visibleRGB], name='X')

Y = tf.placeholder("float",[None, n_visibleRGB], name='X')
#mask = tf.placeholder("float",[None, n_visible], name='mask')

# create hidden var. nodes
if(1):
    W_init_max = 4 * np.sqrt(1. / (n_visibleRGB +n_hidden))
    W_init = tf.random_uniform(shape=[n_visibleRGB,n_hidden],
                               minval=-W_init_max,
                               maxval=W_init_max)
    W_init2 = tf.random_uniform(shape=[n_hidden,n_hidden],
                               minval=-W_init_max,
                               maxval=W_init_max)
    W_init3 = tf.random_uniform(shape=[n_hidden,n_hidden],
                               minval=-W_init_max,
                               maxval=W_init_max)
    W_init4 = tf.random_uniform(shape=[n_hidden,n_hidden],
                               minval=-W_init_max,
                               maxval=W_init_max)
    W_init5 = tf.random_uniform(shape=[n_hidden,n_visibleRGB],
                               minval=-W_init_max,
                               maxval=W_init_max)

    W1 = tf.Variable(W_init, name='W1')
    
    W2 = tf.Variable(W_init2, name='W2')
    print(W2.shape)
    W3 = tf.Variable(W_init3, name='W3')
    print(W3.shape)
    
    W4 = tf.Variable(W_init4, name='W4')
    
    print(W4.shape)
    b = tf.Variable(tf.zeros([n_hidden]),name='b')

    W5 = tf.Variable(W_init5,name = 'W5')
    print(W5.shape)
    b_prime = tf.Variable(tf.zeros([n_visibleRGB]), name='b_prime')

def model(X, Y, W1,W2,W3,W4,W5, b, b_prime):
    tilde_X = X #
    H1 = tf.nn.relu(tf.matmul(tilde_X, W1) + b)
    print(H1.shape)
    H2 = tf.nn.relu(tf.matmul(H1,W2)+b)
    H3 = tf.nn.relu(tf.matmul(H2,W3)+b)
    H4 = tf.nn.relu(tf.matmul(H3,W4)+b)
    #print(np.shape(Y))
    
    #print(np.shape(dO1))
    Z = (tf.matmul(H4,W5) + b_prime)
    
    #Z = (tf.matmul(dO1,W_prime) + b_prime)
    return Z

Z = model(X, Y, W1,W2,W3,W4,W5, b, b_prime)

cost = tf.reduce_sum(tf.pow(Y - Z, 2))
if (0):
    train_op = tf.train.RMSPropOptimizer(learning_rate = lR,
                                         decay=0.9,
                                         momentum=0.0,
                                         epsilon=1e-10,
                                         use_locking=False,
                                         centered=False,
                                         name='RMSProp').minimize(cost)
#train_op = tf.train.GradientDescentOptimizer(lR).minimize(cost)

train_op = tf.train.AdamOptimizer(learning_rate=lR,beta1=0.9,
                                  beta2 = 0.999,epsilon=1e-08,
                                  use_locking=False,
                                  name='Adam').minimize(cost,
                                                        global_step = tf.contrib.framework.get_global_step())

#trX, teX = np.reshape(myImgs[0:800,:,:,0],[np.shape(myImgs[0:800,:,:,0])[0], n_visible]),np.reshape(myImgs[801:1023,:,:,0],[np.shape(myImgs[801:1023:,:,0])[0],n_visible])

#Load data
myTgts = np.load('./simCVTgts/simCVTgts.npy')
myImgs = np.load('./simCVImgs/simCVImgs.npy')

trX = np.reshape(myImgs[0:800,:,:,:],[np.shape(myImgs[0:800,:,:,:])[0],n_visibleRGB])
teX = np.reshape(myImgs[801:923,:,:,:],[np.shape(myImgs[801:923:,:,:])[0],n_visibleRGB])
tcvX = np.reshape(myImgs[924:1023,:,:,:],[np.shape(myImgs[924:1023:,:,:])[0],n_visibleRGB])


trY = np.reshape(myTgts[0:800,:,:,:],[np.shape(myTgts[0:800,:,:,:])[0],n_visibleRGB])
teY = np.reshape(myTgts[801:923,:,:,:],[np.shape(myTgts[801:923:,:,:])[0],n_visibleRGB])
tcvY = np.reshape(myTgts[924:1023,:,:,:],[np.shape(myTgts[924:1023:,:,:])[0],n_visibleRGB])


with tf.Session() as sess:
    #tf.global_variables_initializer
    tf.initialize_all_variables().run() # deprecated, but tf.global_variables_initializer doesn't work at all
    
    for i in range(myIter):
        for start, end in zip(range(0, len(trX), 128),
                              range(128, len(trX), 128)):
            input_ = trX[start:end]
            targets_ = trX[start:end]
            #mask_np = np.random.binomial(1,1-corruption_level, input_.shape)
            sess.run(train_op, feed_dict = {X: input_, Y: targets_})
        #mask_np = np.random.binomial(1,1-corruption_level, teX.shape)
            lR = lR*0.9999
        if( (i) % dispIt == 0):
            print("Epoch %i with training cost %.4e and cross-validation cost %.4e " % (i,sess.run(cost, feed_dict={X: trX, Y: trY}), sess.run(cost, feed_dict={X: tcvX, Y: tcvY}) ))
            print("learning rate decayed to %e" %(lR))
            myElapsed = time.time()-t
            print("elapsed time ", myElapsed, " s")
            
    if(1):
          test_xs = tcvX[0:10,:]
          #recon = sess.run(ae['y'], feed_dict={ae['x']: test_xs_norm})
          #mask_np = np.random.binomial(1,1, test_xs.shape)
          recon = sess.run(Z,feed_dict = {X: test_xs})
          print(recon.shape)
          print(test_xs.shape)

          fig, axs = plt.subplots(3, 10, figsize=(10, 2),dpi=80)
          print("Targets subset >")
          for example_i in range(10):
                
               axs[0][example_i].imshow(np.reshape(test_xs[example_i, :], (imgWidth,imgHeight,3)),cmap="gray")
               axs[0][example_i].set_xticklabels([])
               axs[0][example_i].set_yticklabels([])
               axs[1][example_i].imshow(np.reshape(recon[example_i, ...], (imgWidth,imgHeight,3)),cmap="gray")
               axs[1][example_i].set_xticklabels([])
               axs[1][example_i].set_yticklabels([])
               axs[2][example_i].imshow(np.reshape(tcvY[example_i, ...], (imgWidth,imgHeight,3)),cmap="gray")
               axs[2][example_i].set_xticklabels([])
               axs[2][example_i].set_yticklabels([])
          plt.show()
    print("Guesses subset ^^")
    #myElapsed = time.time()-t
    #print("elapsed time ", myElapsed, " s")
    print("Final Epoch %i with training cost %.4e and cross-validation cost %.4e " % (i,sess.run(cost, feed_dict={X: trX, Y: trY}), sess.run(cost, feed_dict={X: tcvX, Y: tcvY}) ))
    print("Final learning rate decayed to %e" %(lR))
    myElapsed = time.time()-t
    print("elapsed time ", myElapsed, " s")
    print("Finished...")
    
