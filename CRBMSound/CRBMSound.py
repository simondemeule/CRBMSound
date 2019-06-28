import math
import numpy as np
import scipy as sp
import scipy.io.wavfile
import tensorflow as tf
import matplotlib.pyplot as plt

import xrbm.models
import xrbm.train
import xrbm.losses

MODEL_HAAR_ENABLE = False # unimplemented
MODEL_ORDER = 1000
MODEL_NUM_HID = 200
MODEL_LEARN_RATE = 0.01
MODEL_BATCH_SIZE = 2000
MODEL_EPOCHS = 100
MODEL_GIBBS_TRAIN = 1
MODEL_GIBBS_GENERATE = 2
MODEL_CLIP_ENABLE = False # unimplemented
MODEL_CLIP_WINDOW = 0.6

rate, data = sp.io.wavfile.read("input.wav")

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

if MODEL_HAAR_ENABLE:
    print("unimplemented")
else:
    for i in range(MODEL_ORDER, len(data_normalized) - 1):
        condition_data.append(data_normalized[i - MODEL_ORDER: i])
        visible_data.append([data_normalized[i]])

condition_data = np.asarray(condition_data)
visible_data = np.asarray(visible_data)

MODEL_NUM_VIS         = visible_data.shape[1]
MODEL_NUM_COND        = condition_data.shape[1]

# reset the tensorflow graph in case we want to rerun the code
tf.reset_default_graph()

# create CRBM
crbm = xrbm.models.CRBM(num_vis = MODEL_NUM_VIS,
                        num_cond = MODEL_NUM_COND,
                        num_hid = MODEL_NUM_HID,
                        vis_type = 'gaussian',
                        initializer = tf.contrib.layers.xavier_initializer(),
                        name='crbm')

# create mini-batches
batch_indexes = np.random.permutation(range(len(visible_data)))
batch_number  = len(batch_indexes) // MODEL_BATCH_SIZE

# create placeholder
batch_visible_data     = tf.placeholder(tf.float32, shape = (None, MODEL_NUM_VIS), name = 'vis_data')
batch_condition_data   = tf.placeholder(tf.float32, shape = (None, MODEL_NUM_COND), name = 'cond_data')
momentum               = tf.placeholder(tf.float32, shape = ())

# define training operator
cdapproximator     = xrbm.train.CDApproximator(learning_rate = MODEL_LEARN_RATE,
                                               momentum = momentum,
                                               k = MODEL_GIBBS_TRAIN)

train_op           = cdapproximator.train(crbm, vis_data = batch_visible_data, in_data = [batch_condition_data])

reconstructed_data,_,_,_ = crbm.gibbs_sample_vhv(batch_visible_data, [batch_condition_data])
xentropy_rec_cost  = xrbm.losses.cross_entropy(batch_visible_data, reconstructed_data)

# run!
sess = tf.Session()
sess.run(tf.global_variables_initializer())

for epoch in range(MODEL_EPOCHS):

    if epoch < 5: # for the first 5 epochs, we use a momentum coeficient of 0
        epoch_momentum = 0
    else: # once the training is stablized, we use a momentum coeficient of 0.9
        epoch_momentum = 0.9

    for batch_i in range(batch_number):
        # Get just minibatch amount of data
        indexes_i = batch_indexes[batch_i * MODEL_BATCH_SIZE:(batch_i + 1) * MODEL_BATCH_SIZE]

        feed = {batch_visible_data: visible_data[indexes_i],
                batch_condition_data: condition_data[indexes_i],
                momentum: epoch_momentum}

        # Run the training step
        sess.run(train_op, feed_dict=feed)
        # print("Batch %i / %i" % (batch_i + 1, batch_number))

    reconstruction_cost = sess.run(xentropy_rec_cost, feed_dict = feed)


    print('Epoch %i / %i | Reconstruction Cost = %f' %
          (epoch + 1, MODEL_EPOCHS, reconstruction_cost))

# define generator
def generate(crbm, gen_init_frame = 0, num_gen = MODEL_ORDER):
    print('Generating %d frames: ' % (num_gen))

    gen_sample = []
    gen_hidden = []
    initcond = []

    gen_cond = tf.placeholder(tf.float32, shape = [1, MODEL_NUM_COND], name = 'gen_cond_data')
    gen_init = tf.placeholder(tf.float32, shape = [1, MODEL_NUM_VIS], name = 'gen_init_data')
    gen_op = crbm.predict(gen_cond, gen_init, MODEL_GIBBS_GENERATE)

    if MODEL_HAAR_ENABLE:
        print("unimplemented")
    else:
        for f in range(MODEL_ORDER):
            gen_sample.append(np.reshape(visible_data[gen_init_frame + f], [1, MODEL_NUM_VIS]))

        for f in range(num_gen):
            initcond = np.asarray([gen_sample[s] for s in range(f, f + MODEL_ORDER)]).ravel()

            initframes = gen_sample[f + MODEL_ORDER - 1]

            feed = {gen_cond: np.reshape(initcond, [1, MODEL_NUM_COND]).astype(np.float32),
                    gen_init: initframes}

            s, h = sess.run(gen_op, feed_dict=feed)

            """
            # scale normalized to [filerangelow, filerangehigh]
            s[0] = s[0] * data_std + data_mean

            # scale [filerangelow, filerangehigh] to [-1, 1]
            s[0] = s[0] / 32767.0

            # soft clip
            s[0] = softclip(s[0], MODEL_CLIP_WINDOW)

            # scale [-1, 1] to [filerangelow, filerangehigh]
            s[0] = s[0] * 32767.0

            # scale [filerangelow, filerangehigh] to normalized
            s[0] = (s[0] - data_mean) / data_std
            """

            gen_sample.append(s)
            gen_hidden.append(h)

        gen_sample = np.reshape(np.asarray(gen_sample), [num_gen + MODEL_ORDER, MODEL_NUM_VIS])
        gen_hidden = np.reshape(np.asarray(gen_hidden), [num_gen, MODEL_NUM_HID])

        gen_sample = gen_sample * data_std + data_mean

    print("Generation successful")

    return gen_sample, gen_hidden

def generate_to_file(num_gen, gen_init_frame = 0):
    gen_sample, gen_hidden = generate(crbm, gen_init_frame = gen_init_frame, num_gen = num_gen)

    plt.figure(figsize=(12, 6))
    plt.plot(gen_sample)
    plt.title('The Generated Timeseries')
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.imshow(gen_hidden.T, cmap='gray', interpolation='nearest', aspect='auto')
    plt.title('Hidden Units Activities')
    plt.show()

    data_out = [max(min(s[0] / 32767.0, 1.0), -1.0) for s in gen_sample[0:num_gen]]
    data_out = np.asarray(data_out)

    sp.io.wavfile.write("output.wav", rate, data_out)

    plt.figure(figsize=(12, 6))
    plt.plot(data_out)
    plt.title('The Output Timeseries')
    plt.show()

generate_to_file(10000, 0)