# coding=utf-8
import os
import math
import numpy as np
import scipy as sp
import scipy.io.wavfile
import tensorflow as tf
import matplotlib.pyplot as plt

import xrbm.models
import xrbm.train
import xrbm.losses

class model:
    pass

model.batch_size = 1000
model.epochs = 100
model.gibbs_generate = 2
model.gibbs_train = 1
model.input_file = "inputpluck.wav"
model.learn_rate = 0.01
model.num_vis = 1024
model.num_hid = 1024
model.num_cond = 3

rate, data = sp.io.wavfile.read("in/" + model.input_file)

data_max_value = 32767
data_min_value = -32768
data_mean = np.mean(data, axis = 0)
data_std = np.std(data, axis = 0)

def range_data_to_normalized(activation):
    return (activation * 1.0 - data_mean) / data_std

def range_normalized_to_data(activation):
    return (activation * 1.0) * data_std + data_mean

def range_data_to_one(activation):
    return (activation * 1.0 - data_min_value) * 2.0 / (data_max_value - data_min_value) - 1.0

def range_one_to_data(activation):
    return (activation * 1.0 + 1.0) / 2.0 * (data_max_value - data_min_value) + data_min_value

def range_normalized_to_one(activation):
    return range_data_to_one(range_normalized_to_data(activation))

def range_one_to_normalized(activation):
    return range_data_to_normalized(range_one_to_data(activation))

plt.figure(figsize=(12, 6))
plt.plot(data)
plt.title('The Training Sound')
plt.show()


data_one = [range_data_to_one(d) for d in data]
data_fourier = []

# TODO get fourier transform going with appropriate offset, windowing, etc
# TODO also get the reverse operation right for generation

# prep training data
condition_data = []
visible_data = []

for i in range(model.num_cond, len(data_normalized) - 1):
    condition_data.append(data_normalized[i - model.num_cond: i])
    visible_data.append([data_normalized[i]])

condition_data = np.asarray(condition_data)
visible_data = np.asarray(visible_data)

# for a corresponding pair of cond and visible data
# cond data samples    [t - model.num_cond, t - 1] in this order
# visible data samples [t]

# reset the tensorflow graph in case we want to rerun the code
tf.reset_default_graph()

# create CRBM
crbm = xrbm.models.CRBM(num_vis = model.num_vis,
                        num_cond = model.num_cond,
                        num_hid = model.num_hid,
                        vis_type = 'gaussian',
                        initializer = tf.contrib.layers.xavier_initializer(),
                        name='crbm')

# create mini-batches
batch_indexes = np.random.permutation(range(len(visible_data)))
batch_number  = len(batch_indexes) // model.batch_size

# create placeholder
batch_visible_data     = tf.placeholder(tf.float32, shape = (None, model.num_vis), name = 'vis_data')
batch_condition_data   = tf.placeholder(tf.float32, shape = (None, model.num_cond), name = 'cond_data')
momentum               = tf.placeholder(tf.float32, shape = ())

# define training operator
cdapproximator     = xrbm.train.CDApproximator(learning_rate = model.learn_rate,
                                               momentum = momentum,
                                               k = model.gibbs_train)

train_op           = cdapproximator.train(crbm, vis_data = batch_visible_data, in_data = [batch_condition_data])

reconstructed_data,_,_,_ = crbm.gibbs_sample_vhv(batch_visible_data, [batch_condition_data])
xentropy_rec_cost  = xrbm.losses.cross_entropy(batch_visible_data, reconstructed_data)

# run!
sess = tf.Session()
sess.run(tf.global_variables_initializer())

for epoch in range(model.epochs):

    if epoch < 5: # for the first 5 epochs, we use a momentum coeficient of 0
        epoch_momentum = 0
    else: # once the training is stablized, we use a momentum coeficient of 0.9
        epoch_momentum = 0.9

    for batch_i in range(batch_number):
        # Get just minibatch amount of data
        indexes_i = batch_indexes[batch_i * model.batch_size:(batch_i + 1) * model.batch_size]

        feed = {batch_visible_data: visible_data[indexes_i],
                batch_condition_data: condition_data[indexes_i],
                momentum: epoch_momentum}

        # Run the training step
        sess.run(train_op, feed_dict=feed)
        # print("Batch %i / %i" % (batch_i + 1, batch_number))

    reconstruction_cost = sess.run(xentropy_rec_cost, feed_dict = feed)


    print('Epoch %i / %i | Reconstruction Cost = %f' %
          (epoch + 1, model.epochs, reconstruction_cost))

# define generator
def generate(crbm, gen_init_frame = 0, num_gen = model.num_cond):
    print('Generating %d frames: ' % (num_gen))

    gen_sample = []
    gen_hidden = []
    initcond = []

    gen_cond = tf.placeholder(tf.float32, shape = [1, model.num_cond], name = 'gen_cond_data')
    gen_init = tf.placeholder(tf.float32, shape = [1, model.num_vis], name = 'gen_init_data')
    gen_op = crbm.predict(gen_cond, gen_init, model.gibbs_generate)

    # initialization for visible units
    for f in range(model.num_cond):
        gen_sample.append(np.reshape(visible_data[gen_init_frame + f], [1, model.num_vis]))

    for f in range(num_gen):
        # initialization for conditional units
        initcond = np.asarray([gen_sample[s] for s in range(f, f + model.num_cond)]).ravel()

        # initialization for visible units
        initframes = gen_sample[f + model.num_cond - 1]

        # run prediction
        feed = {gen_cond: np.reshape(initcond, [1, model.num_cond]).astype(np.float32),
                gen_init: initframes}

        s, h = sess.run(gen_op, feed_dict=feed)

        gen_sample.append(s)
        gen_hidden.append(h)

    gen_sample = np.reshape(np.asarray(gen_sample), [num_gen + model.num_cond, model.num_vis])
    gen_hidden = np.reshape(np.asarray(gen_hidden), [num_gen, model.num_hid])

    gen_sample = range_normalized_to_data(gen_sample)

    print("Generation successful")

    return gen_sample, gen_hidden

def generate_to_file(num_gen, gen_init_frame = 0):
    gen_sample, gen_hidden = generate(crbm, gen_init_frame = gen_init_frame, num_gen = num_gen)

    plt.figure(figsize=(12, 6))
    plt.plot(gen_sample)
    plt.title('The Generated Timeseries')
    plt.show()

    # TODO: FFT stuff

    # get proper data range
    data_out = [max(min(range_data_to_one(s[0]), 1.0), -1.0) for s in gen_sample[0:num_gen]]
    data_out = np.asarray(data_out)

    # find next avaliable file number
    i = 0
    while os.path.exists("out/output%s.wav" % i) or os.path.exists("output%s.txt" % i):
        i += 1

    # write wav file
    sp.io.wavfile.write("out/output%s.wav" % i, rate, data_out)

    # write txt file
    text_out = open("out/output%s.txt" % i, "w")
    text_out.write('\n'.join("%s: %s" % item for item in sorted(vars(model).items()) if not item[0].startswith('__')))
    text_out.close()

    print("Successfully created files with index %s" % i)

    plt.figure(figsize=(12, 6))
    plt.imshow(gen_hidden.T, cmap='gray', interpolation='nearest', aspect='auto')
    plt.title('Hidden Units Activities')
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.plot(data_out)
    plt.title('The Output Timeseries')
    plt.show()

"""
sess.run(tf.assign(crbm.W, [[0, 0, 0, 0, 0]]))
sess.run(tf.assign(crbm.A, [[0], [0], [0], [0], [1]]))
sess.run(tf.assign(crbm.B, [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]))
sess.run(tf.assign(crbm.vbias, [0]))                    
sess.run(tf.assign(crbm.hbias, [0, 0, 0, 0, 0]))

W = sess.run(crbm.W)
A = sess.run(crbm.A)
B = sess.run(crbm.B)
vbias = sess.run(crbm.vbias)
hbias = sess.run(crbm.hbias)
"""