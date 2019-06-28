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

model.batch_size = 2000
model.clip_enable = False # unimplemented
model.clip_window = 0.6
model.epochs = 100
model.gibbs_generate = 2
model.gibbs_train = 1
model.haar_enable = False # unimplemented
model.input_file = "inputmetal.wav"
model.learn_rate = 0.01
model.num_hid = 200
model.order = 100

rate, data = sp.io.wavfile.read(model.input_file)

"""
def range_data_to_normalized(activations):


def range_normalized_to_data(activations):


def range_normalized_to_one(activations):


def range_one_to_normalized(activations):


# soft clips an input to a range [-1, 1] with a piecewise combination of a sigmoid and a linear region spanning [-window, window]
def softclip(input, window):
    if(-window < input < window):
        return input
    else:
        def sigmoid(x):
            return 2.0 / (1.0 + math.exp(- 2.0 * x)) - 1.0

        a = 1.0 / (1.0 - window)

        if(input > 0):
            return sigmoid(a * (input - window)) / a + window
        else:
            return sigmoid(a * (input + window)) / a - window
"""

plt.figure(figsize=(12, 6))
plt.plot(data)
plt.title('The Training Sound')
plt.show()

# normalize with variance 1, mean 0
data_mean = np.mean(data, axis = 0)
data_std = np.std(data, axis = 0)

data_normalized = [(d - data_mean) / data_std for d in data]

# prep training data
condition_data = []
visible_data = []

# indexing forwards from 1
def haar_sum_upper_bound(n):
    # Sum[2^j, {j, 0, n}] - 1 = 2^(n + 1) - 2
    return pow(2, n + 1) - 2

# indexing forwards from 1
def haar_sum_lower_bound(n):
    # Sum[2^j, {j, 0, n-1}] = 2^n - 1
    return pow(2, n) - 1

# indexing backwards from i
def haar_sum_lower_bound_from(n, i):
    return i - haar_sum_upper_bound(n) - 1

# indexing backwards from i
def haar_sum_upper_bound_from(n, i):
    return i - haar_sum_lower_bound(n) - 1

def haar_sum_num_elements(n):
    return pow(2, n)

# compute visible and conditional unit activations for each timestep for training
if model.haar_enable:
    # using haar sampling
    condition_i_last = []
    for i in range(haar_sum_num_elements(model.order), len(data_normalized) - 1):
        condition_i = []
        if len(condition_i_last) == 0:
            # compute first set of haar samples
            for j in range(model.order):
                sum_j = 0
                for k in range(haar_sum_lower_bound_from(j, i), haar_sum_upper_bound_from(j, i) + 1):
                    sum_j = sum_j + data_normalized[k]
                sum_j = sum_j / haar_sum_num_elements(j)
                condition_i.append(sum_j)
        else:
            # compute later sets with regards to the previous samples
            for j in range(model.order):
                condition_i.append(condition_i_last[j] + (data_normalized[haar_sum_lower_bound_from(j + 1, i)] - data_normalized[haar_sum_lower_bound_from(j, i)]) / haar_sum_num_elements(j))

        condition_i_last = condition_i
        condition_data.append(condition_i)
        visible_data.append([data_normalized[i]])
else:
    # using standard sampling
    for i in range(model.order, len(data_normalized) - 1):
        condition_data.append(data_normalized[i - model.order: i])
        visible_data.append([data_normalized[i]])

print("breakpoint")

condition_data = np.asarray(condition_data)
visible_data = np.asarray(visible_data)

MODEL_NUM_VIS         = visible_data.shape[1]
MODEL_NUM_COND        = condition_data.shape[1]

# reset the tensorflow graph in case we want to rerun the code
tf.reset_default_graph()

# create CRBM
crbm = xrbm.models.CRBM(num_vis = MODEL_NUM_VIS,
                        num_cond = MODEL_NUM_COND,
                        num_hid = model.num_hid,
                        vis_type = 'gaussian',
                        initializer = tf.contrib.layers.xavier_initializer(),
                        name='crbm')

# create mini-batches
batch_indexes = np.random.permutation(range(len(visible_data)))
batch_number  = len(batch_indexes) // model.batch_size

# create placeholder
batch_visible_data     = tf.placeholder(tf.float32, shape = (None, MODEL_NUM_VIS), name = 'vis_data')
batch_condition_data   = tf.placeholder(tf.float32, shape = (None, MODEL_NUM_COND), name = 'cond_data')
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
def generate(crbm, gen_init_frame = 0, num_gen = model.order):
    print('Generating %d frames: ' % (num_gen))

    gen_sample = []
    gen_hidden = []
    initcond = []

    gen_cond = tf.placeholder(tf.float32, shape = [1, MODEL_NUM_COND], name = 'gen_cond_data')
    gen_init = tf.placeholder(tf.float32, shape = [1, MODEL_NUM_VIS], name = 'gen_init_data')
    gen_op = crbm.predict(gen_cond, gen_init, model.gibbs_generate)


    for f in range(model.order):
        gen_sample.append(np.reshape(visible_data[gen_init_frame + f], [1, MODEL_NUM_VIS]))

    for f in range(num_gen):
        # initialization for conditional units
        if model.haar_enable:
            # using haar sampling
            # TODO
            print("unimplemented")
        else:
            # using standard sampling
            initcond = np.asarray([gen_sample[s] for s in range(f, f + model.order)]).ravel()

        # initialization for visible units
        initframes = gen_sample[f + model.order - 1]

        # run prediction
        feed = {gen_cond: np.reshape(initcond, [1, MODEL_NUM_COND]).astype(np.float32),
                gen_init: initframes}

        s, h = sess.run(gen_op, feed_dict=feed)

        """
        # scale normalized to [filerangelow, filerangehigh]
        s[0] = s[0] * data_std + data_mean

        # scale [filerangelow, filerangehigh] to [-1, 1]
        s[0] = s[0] / 32767.0

        # soft clip
        s[0] = softclip(s[0], model.clip_window)

        # scale [-1, 1] to [filerangelow, filerangehigh]
        s[0] = s[0] * 32767.0

        # scale [filerangelow, filerangehigh] to normalized
        s[0] = (s[0] - data_mean) / data_std
        """

        gen_sample.append(s)
        gen_hidden.append(h)

        gen_sample = np.reshape(np.asarray(gen_sample), [num_gen + model.order, MODEL_NUM_VIS])
        gen_hidden = np.reshape(np.asarray(gen_hidden), [num_gen, model.num_hid])

        gen_sample = gen_sample * data_std + data_mean

    print("Generation successful")

    return gen_sample, gen_hidden

def generate_to_file(num_gen, gen_init_frame = 0):
    gen_sample, gen_hidden = generate(crbm, gen_init_frame = gen_init_frame, num_gen = num_gen)

    plt.figure(figsize=(12, 6))
    plt.plot(gen_sample)
    plt.title('The Generated Timeseries')
    plt.show()

    # get proper data range
    data_out = [max(min(s[0] / 32767.0, 1.0), -1.0) for s in gen_sample[0:num_gen]]
    data_out = np.asarray(data_out)

    # find next avaliable file number
    i = 0
    while os.path.exists("output%s.wav" % i) or os.path.exists("output%s.txt" % i):
        i += 1

    # write wav file
    sp.io.wavfile.write("output%s.wav" % i, rate, data_out)

    # write txt file
    text_out = open("output%s.txt" % i, "w")
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

generate_to_file(10000, 0)