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
model.clip_enable = False
model.clip_window = 0.8
model.epochs = 30
model.gibbs_generate = 2
model.gibbs_train = 1
model.input_file = "drumsamekick/drumsamekick1.wav"
model.learn_rate = 0.01
model.multiscale_base = 2
model.multiscale_enable = False
model.num_hid = 50
model.num_cond = 50

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

plt.figure(figsize=(12, 6))
plt.plot(data)
plt.title('The Training Sound')
plt.show()


data_normalized = [range_data_to_normalized(d) for d in data]
data_normalized_sigmoid = []
if model.clip_enable:
    data_normalized_sigmoid = [range_one_to_normalized(softclip(range_normalized_to_one(d), model.clip_window)) for d in data_normalized]

# prep training data
condition_data = []
visible_data = []

# TODO: either trash this or fix it. Super broken at the moment. The first 4 functions are fine, but the data prep / generation routine are completely broken and need to be rewritten and tested.

"""
def multiscale_size(n):
    return int(math.ceil(pow(model.multiscale_base, n)))

multiscale_offset_computed = [0]

def multiscale_offset(n):
    if n <= len(multiscale_offset_computed) - 1:
        # pre-computed result already exists
        return multiscale_offset_computed[n]
    else:
        # compute result recursively
        next = multiscale_offset_computed[len(multiscale_offset_computed) - 1] + multiscale_size(len(multiscale_offset_computed) - 1)
        multiscale_offset_computed.append(next)
        return multiscale_offset(n)

def multiscale_total_size():
    return multiscale_offset(model.num_cond)

def multiscale_range_reverse_from(n, i):
    # i is the excluded visible unit. first conditional unit contains i - 1
    return range((i - 1) - (multiscale_offset(n) + multiscale_size(n) - 1),(i - 1) - multiscale_offset(n) + 1)

if model.multiscale_enable:
    for i in range(model.num_cond):
        print("unit %i\tsize %i\toffset %i" % (i, multiscale_size(i), multiscale_offset(i)))
        print(multiscale_range_reverse_from(i, 0))

# compute visible and conditional unit activations for each timestep for training
if model.multiscale_enable:
    # using multiscale sampling
    condition_i_last = []
    for i in range(multiscale_total_size() + 1, len(data_normalized)):
        condition_i = []
        if len(condition_i_last) == 0:
            # compute first set of multiscale samples
            for n in range(model.num_cond):
                sum_n = 0
                for k in multiscale_range_reverse_from(n, i):
                    # apply softclip if needed
                    if model.clip_enable:
                        sum_n = sum_n + data_normalized_sigmoid[k]
                    else:
                        sum_n = sum_n + data_normalized[k]
                sum_n = sum_n / multiscale_sum_num_elements(n)
                condition_i.append(sum_n)
        else:
            # compute later sets with regards to the previous samples
            for n in range(model.num_cond):
                if model.clip_enable:
                    condition_i.append(condition_i_last[n] + (data_normalized_sigmoid[multiscale_sum_lower_bound_from(n + 1, i)] - data_normalized_sigmoid[multiscale_sum_lower_bound_from(n, i)]) / multiscale_sum_num_elements(n))
                else:
                    condition_i.append(condition_i_last[n] + (data_normalized[multiscale_sum_lower_bound_from(n + 1, i)] - data_normalized[multiscale_sum_lower_bound_from(n, i)]) / multiscale_sum_num_elements(n))
        condition_i_last = condition_i
        condition_data.append(condition_i)
        visible_data.append([data_normalized[i]])
else:
"""
# using standard sampling
for i in range(model.num_cond, len(data_normalized) - 1):
    # apply softclip if needed
    if model.clip_enable:
        condition_data.append(data_normalized_sigmoid[i - model.num_cond: i])
    else:
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
crbm = xrbm.models.CRBM(num_vis = 1,
                        num_cond = model.num_cond,
                        num_hid = model.num_hid,
                        vis_type = 'gaussian',
                        initializer = tf.contrib.layers.xavier_initializer(),
                        name='crbm')

# create mini-batches
batch_indexes = np.random.permutation(range(len(visible_data)))
batch_number  = len(batch_indexes) // model.batch_size

# create placeholder
batch_visible_data     = tf.placeholder(tf.float32, shape = (None, 1), name = 'vis_data')
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
    gen_init = tf.placeholder(tf.float32, shape = [1, 1], name = 'gen_init_data')
    gen_op = crbm.predict(gen_cond, gen_init, model.gibbs_generate)
    """
    if model.multiscale_enable:
        # using multiscale sampling

        # first initialization for samples
        for f in range(multiscale_sum_num_elements_integral(model.num_cond - 1)):
            gen_sample.append(np.reshape(visible_data[gen_init_frame + f], [1, 1]))

        for f in range(num_gen):
            if f == 0:
                # first initialization for conditional units
                a = gen_init_frame
                initcond = np.asarray(condition_data[a])

                # first initialization for visible unit (copy of previous value)
                b = gen_init_frame + multiscale_sum_num_elements_integral(model.num_cond - 1) - 1
                initframes = gen_sample[b]
            else:
                # recursive initialization for conditional units
                # TODO: does condition_data[i] correspond to order i or model.num_cond - i ?
                for j in range(model.num_cond):
                    a = multiscale_sum_lower_bound_from(j + 1, f + multiscale_sum_num_elements_integral(model.num_cond - 1))
                    b = multiscale_sum_lower_bound_from(j, f + multiscale_sum_num_elements_integral(model.num_cond - 1))
                    initcond[j] = (initcond[j] + (gen_sample[a] - gen_sample[b]) / multiscale_sum_num_elements(j))


                # recursive initialization for visible unit (copy of previous value)
                i = f + multiscale_sum_num_elements_integral(model.num_cond - 1) - 1
                initframes = gen_sample[i]

            # run prediction
            feed = {gen_cond: np.reshape(initcond, [1, model.num_cond]).astype(np.float32),
                    gen_init: initframes}

            s, h = sess.run(gen_op, feed_dict=feed)

            if model.clip_enable:
                s[0] = range_one_to_normalized(softclip(range_normalized_to_one(s[0]), model.clip_window))

            gen_sample.append(s)
            gen_hidden.append(h)

        gen_sample = np.reshape(np.asarray(gen_sample), [num_gen + multiscale_sum_num_elements_integral(model.num_cond - 1), 1])
        gen_hidden = np.reshape(np.asarray(gen_hidden), [num_gen, model.num_hid])
    else:
    """
    # using standard sampling

    # initialization for visible units
    for f in range(model.num_cond):
        gen_sample.append(np.reshape(visible_data[gen_init_frame + f], [1, 1]))

    for f in range(num_gen):
        # initialization for conditional units
        initcond = np.asarray([gen_sample[s] for s in range(f, f + model.num_cond)])

        # initialization for visible units
        initframes = gen_sample[f + model.num_cond - 1]

        # run prediction
        feed = {gen_cond: np.reshape(initcond, [1, model.num_cond]).astype(np.float32),
                gen_init: initframes}

        s, h = sess.run(gen_op, feed_dict=feed)

        if model.clip_enable:
            s[0] = range_one_to_normalized(softclip(range_normalized_to_one(s[0]), model.clip_window))

        gen_sample.append(s)
        gen_hidden.append(h)

    gen_sample = np.reshape(np.asarray(gen_sample), [num_gen + model.num_cond, 1])
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

    # TODO: double check the proper range of values is being sent to file

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

def a_to_polynomial(a):
    """
    if i = 0
        a_as_polynomial[i] = 1
    if 0 < i < model.num_cond + 1
        a_as_polynomial[i] = - a[model.num_cond - i][0]
    """

    a_as_polynomial = [1 for _ in range(model.num_cond + 1)]

    for i in range(1, model.num_cond + 1):
        a_as_polynomial[i] = - a[model.num_cond - i][0]
    return a_as_polynomial

def a_from_polynomial(a_as_polynomial):
    """
    if 0 < i < model.num_cond
        a[i][0] = - a_as_polynomial[model.num_cond - i]
    """

    a = [[0] for _ in range(model.num_cond)]

    for i in range(model.num_cond):
        a[i][0] = - a_as_polynomial[model.num_cond - i]
    return a

def a_to_poles(a):
    a_as_polynomial = a_to_polynomial(a)
    a_as_poles = np.roots(a_as_polynomial)
    return a_as_poles

def a_from_poles(a_as_poles):
    a_as_polynomial = np.poly(a_as_poles)
    a = a_from_polynomial(a_as_polynomial)
    return a

def b_to_polynomial_array(b):
    # indexing convention
    # b[cond][hid]
    # b_as_polynomial_array[hid][poly]

    b_as_polynomial_array = [[1 for _ in range(model.num_cond + 1)] for _ in range(model.num_hid)]

    for h in range(model.num_hid):
        for i in range(1, model.num_cond + 1):
            b_as_polynomial_array[h][i] = - b[model.num_cond - i][h]
    return b_as_polynomial_array

def b_from_polynomial_array(b_as_polynomial_array):
    # indexing convention
    # b[cond][hid]
    # b_as_polynomial_array[hid][poly]

    b = [[0 for _ in range(model.num_hid)] for _ in range(model.num_cond)]

    for h in range(model.num_hid):
        for i in range(model.num_cond):
            b[i][h] = - b_as_polynomial_array[h][model.num_cond - i]
    return b

def b_to_poles_array(b):
    # indexing convention
    # b[cond][hid]
    # b_as_poles_array[hid][pole]

    b_as_polynomial_array = b_to_polynomial_array(b)
    b_as_poles_array = []
    for b_as_polynomial in b_as_polynomial_array:
        b_as_poles_array.append(np.roots(b_as_polynomial))
    return b_as_poles_array

def b_from_poles_array(b_as_poles_array):
    # indexing convention
    # b[cond][hid]
    # b_as_poles_array[hid][pole]

    b_as_polynomial_array = []
    for b_as_poles in b_as_poles_array:
        b_as_polynomial_array.append(np.poly(b_as_poles))
    b = b_from_polynomial_array(b_as_polynomial_array)
    return b

def s_to_polynomial(a, b, w):
    s_as_polynomial = a_to_polynomial(a)
    b_as_polynomial_array = b_to_polynomial_array(b)
    for h in range(model.num_hid):
        for i in range(1, model.num_cond + 1):
            s_as_polynomial[i] += b_as_polynomial_array[h][i] * w[0][h]
    return s_as_polynomial

def s_to_poles(a, b, w):
    s_as_polynomial = s_to_polynomial(a, b, w)
    s_as_poles = np.roots(s_as_polynomial)
    return s_as_poles


def poles_norm_clamp(poles, epsilon):
    # clamp poles to within a circle of radius 1 - epsilon
    # preserves angle
    poles_result = []
    for pole in poles:
        norm = np.absolute(pole)
        if norm > 1.0 - epsilon:
            factor = (1.0 / norm) * (1.0 - epsilon)
            pole = pole * factor
        poles_result.append(pole)
    return poles_result

def poles_norm_multiply(poles, factor):
    # multiply the pole's norm by the factor
    poles_result = []
    for pole in poles:
        pole = pole * factor
        poles_result.append(pole)
    return poles_result

def poles_norm_pull(poles, factor):
    # divide the distance of the pole to the unit circle by the factor given
    # factor > 1 pulls towards unit circle
    # factor < 1 pulls away from unit circle
    # factor = 2   halves the distance of the pole to the unit circle
    # factor = 1/2 doubles the distance of the pole to the unit circle
    poles_result = []
    for pole in poles:
        norm = np.absolute(pole)
        norm = (1.0 - (1.0 - norm) / factor) / norm
        pole = pole * norm
        poles_result.append(pole)
    return poles_result

def poles_angle_add(poles, theta, epsilon):
    # add to the pole's angle
    # poles with an imaginary component smaller than epsilon are not affected
    # keeps the polynomial real
    poles_result = []
    for pole in poles:
        if abs(pole.imag) > epsilon:
            if (pole.imag > 0):
                pole = pole * np.exp(1j * theta)
            else:
                pole = pole * np.exp(- 1j * theta)
        poles_result.append(pole)
    return poles_result


def poles_angle_multiply(poles, factor, epsilon):
    # TODO: how to deal with warping over nyquist?
    # multiply the pole's angle
    # poles with an imaginary component smaller than epsilon are not affected
    # keeps the polynomial real
    poles_result = []
    for pole in poles:
        if abs(pole.imag) > epsilon:
            flip = pole.imag < 0
            norm = np.absolute(pole)
            if flip:
                pole = np.conjugate(pole)
            angle = np.angle(pole) * factor
            pole = norm * np.exp(1j * angle)
            if flip:
                pole = np.conjugate(pole)
        poles_result.append(pole)
    return poles_result

def plot_all_poles(a, b, w):

    D1 = a_to_poles(a)
    m1 = max(max(abs(D1.real)), max(abs(D1.imag)))

    D2 = []
    for b_as_poles in b_to_poles_array(b):
        for b_pole in b_as_poles:
            D2.append(b_pole)
    D2 = np.asarray(D2)
    m2 = max(max(abs(D2.real)), max(abs(D2.imag)))

    D3 = s_to_poles(a, b, w)
    m3 = max(max(abs(D3.real)), max(abs(D3.imag)))

    #m1 = m2 = m3 = max(m1, m2, m3)
    if 1.5 < m1 or 1.5 < m2 or 1.5 < m3:
        print("Warning: some poles fall out of plot range")
    m1 = m2 = m3 = 1.5

    t = np.linspace(0,2 * np.pi, 101)
    plt.plot(np.cos(t), np.sin(t))
    plt.scatter([x.real for x in D1],[x.imag for x in D1], color='red')
    plt.axes().set_aspect('equal')
    plt.xlim(-1.1 * m1,1.1 * m1)
    plt.ylim(-1.1 * m1,1.1 * m1)
    plt.title('Conditional Poles')
    plt.show()

    t = np.linspace(0,2 * np.pi, 101)
    plt.plot(np.cos(t), np.sin(t))
    plt.scatter([x.real for x in D2],[x.imag for x in D2], color='red')
    plt.axes().set_aspect('equal')
    plt.xlim(-1.1 * m2,1.1 * m2)
    plt.ylim(-1.1 * m2,1.1 * m2)
    plt.title('Hidden Poles')
    plt.show()

    t = np.linspace(0,2 * np.pi, 101)
    plt.plot(np.cos(t), np.sin(t))
    plt.scatter([x.real for x in D3],[x.imag for x in D3], color='red')
    plt.axes().set_aspect('equal')
    plt.xlim(-1.1 * m3,1.1 * m3)
    plt.ylim(-1.1 * m3,1.1 * m3)
    plt.title('Total Poles')
    plt.show()

#def plot_poles(poles, scale):

def plot_polynomial(polynomial):
    freq = np.linspace(- np.pi, np.pi, 1000)
    plane = [np.exp(1j * f) for f in freq]
    response = [1.0 / np.polyval(polynomial, z) for z in plane]
    plt.plot(freq, response)
    plt.xlim(- np.pi, np.pi)
    plt.ylim(-40, 40)
    plt.show()

def plot_two_polynomial(a_as_polynomial, s_as_polynomial):
    freq = np.linspace(- np.pi, np.pi, 1000)
    plane = [np.exp(1j * f) for f in freq]
    a_response = [1.0 / np.polyval(a_as_polynomial, z) for z in plane]
    s_response = [1.0 / np.polyval(s_as_polynomial, z) for z in plane]
    plt.plot(freq, a_response, 'r')
    plt.plot(freq, s_response, 'b')
    plt.xlim(- np.pi, np.pi)
    plt.ylim(-40, 40)
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

A = sess.run(crbm.A)            # idx by cond -> idx by vis
B = sess.run(crbm.B)            # idx by cond -> idx by hid
W = sess.run(crbm.W)            # idx by vis  -> idx by hid
#vbias = sess.run(crbm.vbias)    # idx by vis
#hbias = sess.run(crbm.hbias)    # idx by hid

plot_all_poles(A, B, W)
plot_two_polynomial(a_to_polynomial(A), s_to_polynomial(A, B, W))

model.meta = "original"
generate_to_file(30000, 1000)

A = sess.run(crbm.A)
A_as_poles = a_to_poles(A)
A_as_poles = poles_norm_clamp(A_as_poles, 0.001)
A = a_from_poles(A_as_poles)
sess.run(tf.assign(crbm.A, A))

B = sess.run(crbm.B)
B_as_poles_array = []
for B_as_poles in b_to_poles_array(B):
    B_as_poles = poles_norm_clamp(B_as_poles, 0.001)
    B_as_poles_array.append(B_as_poles)
B = b_from_poles_array(B_as_poles_array)
sess.run(tf.assign(crbm.B, B))

plot_all_poles(A, B, W)
plot_two_polynomial(a_to_polynomial(A), s_to_polynomial(A, B, W))

model.meta = "pole clamp A and B"
generate_to_file(30000, 1000)

sess.run(tf.assign(crbm.W, [[0 for _ in range(model.num_hid)]]))
sess.run(tf.assign(crbm.vbias, [0]))

model.meta = "eliminating hidden contributions and offset"
generate_to_file(30000, 1000)

"""
A = sess.run(crbm.A)
A_as_poles = a_to_poles(A)
A_as_poles = poles_angle_multiply(A_as_poles, 2.0, 0.001)
A = a_from_poles(A_as_poles)
sess.run(tf.assign(crbm.A, A))

B = sess.run(crbm.B)
B_as_poles_array = []
for B_as_poles in b_to_poles_array(B):
    B_as_poles = poles_angle_multiply(B_as_poles, 2.0, 0.001)
    B_as_poles_array.append(B_as_poles)
B = b_from_poles_array(B_as_poles_array)
sess.run(tf.assign(crbm.B, B))

plot_all_poles(A, B, W)
plot_two_polynomial(a_to_polynomial(A), s_to_polynomial(A, B, W))

model.meta = "pole angle multiply by 2"
generate_to_file(30000, 1000)
"""

"""
# numerical stability / precision test for polynomial -> pole -> polynomial conversion
for i in range(100):
    A_as_poles = a_to_poles(A)

    m1 = 1.5

    t = np.linspace(0, 2 * np.pi, 101)
    plt.plot(np.cos(t), np.sin(t))
    plt.scatter([x.real for x in A_as_poles], [x.imag for x in A_as_poles], color='red')
    plt.axes().set_aspect('equal')
    plt.xlim(-1.1 * m1, 1.1 * m1)
    plt.ylim(-1.1 * m1, 1.1 * m1)
    plt.title('Conditional Poles')
    plt.show()

    A = a_from_poles(A_as_poles)
"""