# coding=utf-8
import os
import math
import numpy as np
import scipy as sp
import scipy.io.wavfile
import scipy.signal
import tensorflow as tf
import matplotlib.pyplot as plt

import xrbm.models
import xrbm.train
import xrbm.losses

class model:
    pass

model.batch_size = 32
model.epochs = 100
model.gibbs_generate = 2
model.gibbs_train = 1
model.input_file = "inputcello.wav"
model.learn_rate = 0.01
model.transform_segment = 512
model.transform_window = "hann"
model.num_vis = model.transform_segment + 2
model.num_hid = model.transform_segment + 2
model.num_cond = model.transform_segment + 2

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

def fourier_forward(x):
    _, _, z = scipy.signal.stft(x, fs = rate, nperseg = model.transform_segment, window = model.transform_window)
    return z

def fourier_inverse(z):
    _, x = sp.signal.istft(z, fs = rate, nperseg = model.transform_segment, window = model.transform_window)
    return x

def fourier_to_fourier_delta(z):
    zd = np.empty_like(z)
    for f in range(z.shape[0]):
        angle_prev = z[f][0]
        for t in range(z.shape[1]):
            angle_current = np.angle(z[f][t])
            # calculate phase change
            angle_delta = np.real(angle_current - angle_prev)
            # cyclical unwrap
            angle_delta = (angle_delta + np.pi) % (2.0 * np.pi) - np.pi
            # write new transform values
            zd[f][t] = np.abs(z[f][t]) * np.exp(-1.0j * angle_delta)
            angle_prev = angle_current
    return zd

def fourier_delta_to_fourier(zd, angle_init):
    z = np.empty_like(zd)
    for f in range(zd.shape[0]):
        angle_prev = angle_init[f]
        for t in range(zd.shape[1]):
            angle_delta = np.angle(zd[f][t])
            # apply phase change
            angle_current = angle_prev + angle_delta
            # cyclical wrap
            angle_current = (angle_current + np.pi) % (2.0 * np.pi) - np.pi
            # write new transform values
            z[f][t] = np.abs(zd[f][t]) * np.exp(-1.0j * angle_current)
            angle_prev = angle_current
    return z

def fourier_to_polar_delta(z):
    amplitude = np.empty_like(z)
    angle = np.empty_like(z)
    for f in range(z.shape[0]):
        angle_prev = z[f][0]
        for t in range(z.shape[1]):
            angle_current = np.angle(z[f][t])
            # calculate phase change
            angle_delta = np.real(angle_current - angle_prev)
            # cyclical unwrap
            angle_delta = (angle_delta + np.pi) % (2.0 * np.pi) - np.pi
            # write new transform values
            amplitude[f][t] = np.abs(z[f][t])
            angle[f][t] = angle_delta
            angle_prev = angle_current
    return amplitude, angle

def polar_delta_to_fourier(amplitude, angle, angle_initial):
    z = np.empty_like(amplitude)
    for f in range(amplitude.shape[0]):
        angle_prev = angle_initial[f]
        for t in range(amplitude.shape[1]):
            angle_delta = angle[f][t]
            # apply phase change
            angle_current = angle_prev + angle_delta
            # cyclical wrap
            angle_current = (angle_current + np.pi) % (2.0 * np.pi) - np.pi
            # write new transform values
            z[f][t] = amplitude[f][t] * np.exp(-1.0j * angle_current)
            angle_prev = angle_current
    return z

def polar_to_fourier(amplitude, angle):
    z = np.empty_like(amplitude)
    for f in range(amplitude.shape[0]):
        for t in range(amplitude.shape[1]):
            z[f][t] = amplitude[f][t] * np.exp(-1.0j * angle[f][t])
    return z

def fourier_to_polar(z):
    amplitude = np.empty_like(z)
    angle = np.empty_like(z)
    for f in range(z.shape[0]):
        for t in range(z.shape[1]):
            amplitude[f][t] = np.abs(z[f][t])
            angle[f][t] = np.angle(z[f][t])
    return amplitude, angle

def polar_to_activation_polar(amplitude, angle):
    activation = np.empty((amplitude.shape[1], amplitude.shape[0] * 2))
    for t in range(amplitude.shape[1]):
        for f in range(amplitude.shape[0]):
            activation[t][2 * f] = amplitude[f][t]
            activation[t][2 * f + 1] = angle[f][t]
    return activation

def activation_polar_to_polar(activation):
    amplitude = np.empty((activation.shape[1] / 2, activation.shape[0]))
    angle = np.empty((activation.shape[1] / 2, activation.shape[0]))
    for t in range(amplitude.shape[1]):
        for f in range(amplitude.shape[0]):
            amplitude[f][t] = activation[t][2 * f]
            angle[f][t] = activation[t][2 * f + 1]
    return amplitude, angle

def fourier_to_activation_fourier(z):
    activation = np.empty((z.shape[1], z.shape[0] * 2))
    for t in range(z.shape[1]):
        for f in range(z.shape[0]):
            activation[t][2 * f] = np.real(z[f][t])
            activation[t][2 * f + 1] = np.imag(z[f][t])
    return activation

def activation_fourier_to_fourier(activation):
    z = np.empty((activation.shape[1] / 2, activation.shape[0]))
    for t in range(z.shape[1]):
        for f in range(z.shape[0]):
            z[f][t] = activation[t][2 * f] + 1.0j * activation[t][2 * f + 1]
    return z

def plot_fourier_amplitude(z):
    f = np.asarray([rate * 1.0 / model.transform_segment * i for i in range(0, model.transform_segment / 2 + 1)])
    t = np.asarray([model.transform_segment / 2.0 / rate * i for i in range(z.shape[1])])
    plt.figure()
    plt.pcolormesh(t, f, np.abs(z))
    plt.ylim([f[1], f[-1]])
    plt.title('STFT Magnitude')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.yscale('log')
    plt.colorbar()
    plt.show()

def plot_fourier_phase(z):
    f = np.asarray([rate * 1.0 / model.transform_segment * i for i in range(0, model.transform_segment / 2 + 1)])
    t = np.asarray([model.transform_segment / 2.0 / rate * i for i in range(z.shape[1])])
    plt.figure()
    plt.pcolormesh(t, f, np.angle(z))
    plt.ylim([f[1], f[-1]])
    plt.title('STFT Phase')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.yscale('log')
    plt.colorbar()
    plt.show()

data_fourier = fourier_forward(range_data_to_normalized(data))
data_fourier_delta = fourier_to_fourier_delta(data_fourier)

plot_fourier_amplitude(data_fourier)
plot_fourier_phase(data_fourier)
plot_fourier_phase(data_fourier_delta)

data_activation = fourier_to_activation_fourier(data_fourier)

# TODO load data into condition / visible data
# TODO unload data after generation and compute inverse fft
# TODO add model param for phase unwrap on / off

# prep training data
condition_data = np.empty((data_activation.shape[0] - 1, data_activation.shape[1]))
visible_data = np.empty((data_activation.shape[0] - 1, data_activation.shape[1]))

for t in range(visible_data.shape[0]):
    condition_data[t] = data_activation[t]
    visible_data[t] = data_activation[t + 1]

"""
condition_data = []
visible_data = []


for i in range(model.num_cond, data_fourier.shape[1] - 1):
    condition_data.append(data_activation[i - model.num_cond: i])
    visible_data.append(data_activation[i])

condition_data = np.asarray(condition_data)
visible_data = np.asarray(visible_data)
"""

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

W = sess.run(crbm.W)
A = sess.run(crbm.A)
B = sess.run(crbm.B)
vbias = sess.run(crbm.vbias)
hbias = sess.run(crbm.hbias)

print("done")