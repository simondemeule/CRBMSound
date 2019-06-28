import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import xrbm.models
import xrbm.train
import xrbm.losses

# model related
MODEL_NUM_DIM = 1
MODEL_ORDER = 10
MODEL_NUM_HID         = 20
MODEL_LEARN_RATE   = 0.01
MODEL_BATCH_SIZE      = 50
MODEL_EPOCHS = 10
MODEL_GEN_LENGTH = 400

# signal related
FREQS = [2]
AMPS = [1]
SAMPLE_TIMESPAN = 2
SAMPLE_NUMBER = 100

time_data = np.arange(SAMPLE_NUMBER) / np.float32(SAMPLE_NUMBER) * np.float(SAMPLE_TIMESPAN)

X_train = []

# create time series
for i in range(SAMPLE_NUMBER):
    x = [np.float32(
        np.sin(freq * 2 * np.pi * time_data) * amp + np.sin(freq * 2 * 6 * np.pi * time_data) * amp / 3 + np.sin(freq * 2 * 3 * np.pi * time_data) * amp / 4)
        for freq, amp in zip(FREQS, AMPS)]

    x = np.asarray(x)
    #x = x + np.random.rand(x.shape[0], x.shape[1]) * 0.3

    X_train.append(x.T)

X_train = np.asarray(X_train)

print(X_train.shape)

# plot time series
plt.figure(figsize=(12, 6))
plt.plot(X_train[0,0:len(X_train),:])
plt.title('The Training Timeseries')
plt.show()

# normalize with variance 1, mean 0
X_train_flat = np.concatenate([m for m in X_train], axis=0)
data_mean = np.mean(X_train_flat, axis=0)
data_std = np.std(X_train_flat, axis=0)

X_train_normalized = [(d - data_mean) / data_std for d in X_train]

# prep training data
condition_data = []
visible_data = []

for m in X_train_normalized:
    for i in range(len(m) - MODEL_ORDER):
        condition_data.append(m[i:i + MODEL_ORDER].flatten())
        visible_data.append(m[i + MODEL_ORDER])

condition_data = np.asarray(condition_data)
visible_data = np.asarray(visible_data)

# create CRBM
MODEL_NUM_VIS         = visible_data.shape[1]
MODEL_NUM_COND        = condition_data.shape[1]

# reset the tensorflow graph in case we want to rerun the code
tf.reset_default_graph()

crbm = xrbm.models.CRBM(num_vis=MODEL_NUM_VIS,
                        num_cond=MODEL_NUM_COND,
                        num_hid=MODEL_NUM_HID,
                        vis_type='gaussian',
                        initializer=tf.contrib.layers.xavier_initializer(),
                        name='crbm')

# create mini-batches
batch_idxs = np.random.permutation(range(len(visible_data)))
n_batches  = len(batch_idxs) // MODEL_BATCH_SIZE

# create placeholder
batch_vis_data     = tf.placeholder(tf.float32, shape=(None, MODEL_NUM_VIS), name='batch_data')
batch_cond_data    = tf.placeholder(tf.float32, shape=(None, MODEL_NUM_COND), name='cond_data')
momentum           = tf.placeholder(tf.float32, shape=())

# define training operator
cdapproximator     = xrbm.train.CDApproximator(learning_rate=MODEL_LEARN_RATE,
                                               momentum=momentum,
                                               k=1) # perform 1 step of gibbs sampling

train_op           = cdapproximator.train(crbm, vis_data=batch_vis_data, in_data=[batch_cond_data])

reconstructed_data,_,_,_ = crbm.gibbs_sample_vhv(batch_vis_data, [batch_cond_data])
xentropy_rec_cost  = xrbm.losses.cross_entropy(batch_vis_data, reconstructed_data)

# define generator
def generate(crbm, gen_init_frame=100, num_gen=200):
    gen_sample = []
    gen_hidden = []
    initcond = []

    gen_cond = tf.placeholder(tf.float32, shape=[1, MODEL_NUM_COND], name='gen_cond_data')
    gen_init = tf.placeholder(tf.float32, shape=[1, MODEL_NUM_VIS], name='gen_init_data')
    gen_op = crbm.predict(gen_cond, gen_init, 2)  # 2 stands for the number of gibbs sampling iterations

    for f in range(MODEL_ORDER):
        gen_sample.append(np.reshape(visible_data[gen_init_frame + f], [1, MODEL_NUM_VIS]))

    print('Generating %d frames: ' % (num_gen))

    for f in range(num_gen):
        initcond = np.asarray([gen_sample[s] for s in range(f, f + MODEL_ORDER)]).ravel()

        initframes = gen_sample[f + MODEL_ORDER - 1]

        feed = {gen_cond: np.reshape(initcond, [1, MODEL_NUM_COND]).astype(np.float32),
                gen_init: initframes}

        s, h = sess.run(gen_op, feed_dict=feed)

        gen_sample.append(s)
        gen_hidden.append(h)

    gen_sample = np.reshape(np.asarray(gen_sample), [num_gen + MODEL_ORDER, MODEL_NUM_VIS])
    gen_hidden = np.reshape(np.asarray(gen_hidden), [num_gen, MODEL_NUM_HID])

    gen_sample = gen_sample * data_std + data_mean

    print("Generation successful")

    return gen_sample, gen_hidden

# try the thing
sess = tf.Session()
sess.run(tf.global_variables_initializer())


#     gen_sample, gen_hidden = generate(crbm, num_gen=70)
#     fig = plt.figure(figsize=(12, 3))
#     _ = plt.plot(gen_sample)
#     display.display(fig)

for epoch in range(MODEL_EPOCHS):

    if epoch < 5: # for the first 5 epochs, we use a momentum coeficient of 0
        epoch_momentum = 0
    else: # once the training is stablized, we use a momentum coeficient of 0.9
        epoch_momentum = 0.9

    for batch_i in range(n_batches):
        # Get just minibatch amount of data
        idxs_i = batch_idxs[batch_i * MODEL_BATCH_SIZE:(batch_i + 1) * MODEL_BATCH_SIZE]

        feed = {batch_vis_data: visible_data[idxs_i],
                batch_cond_data: condition_data[idxs_i],
                momentum: epoch_momentum}

        # Run the training step
        sess.run(train_op, feed_dict=feed)

    reconstruction_cost = sess.run(xentropy_rec_cost, feed_dict=feed)


    print('Epoch %i / %i | Reconstruction Cost = %f' %
          (epoch + 1, MODEL_EPOCHS, reconstruction_cost))

gen_sample, gen_hidden = generate(crbm, num_gen = MODEL_GEN_LENGTH) #SAMPLE_NUMBER - MODEL_ORDER

plt.figure(figsize=(12, 6))
plt.plot(gen_sample)
plt.title('The Generated Timeseries')
plt.show()

plt.figure(figsize=(12, 6))
plt.imshow(gen_hidden.T, cmap='gray', interpolation='nearest', aspect='auto')
plt.title('Hidden Units Activities')
plt.show()

W = sess.run(crbm.W)
A = sess.run(crbm.A)
B = sess.run(crbm.B)
vbias = sess.run(crbm.vbias)
hbias = sess.run(crbm.hbias)

print("\n> W\n\n" + W.__str__())
print("\n> A\n\n" + A.__str__())
print("\n> B\n\n" + B.__str__())
print("\n> a\n\n" + vbias.__str__())
print("\n> b\n\n" + hbias.__str__())

"""
plt.figure(figsize=(12, 6))
plt.imshow(W, cmap='gray', interpolation='nearest', aspect='auto')
plt.title('Weight Matix W')
#plt.axis([0, MODEL_NUM_HID, 0, MODEL_NUM_DIM])
#plt.axis('scaled')
plt.show()

plt.figure(figsize=(12, 6))
plt.imshow(A, cmap='gray', interpolation='nearest', aspect='auto')
plt.title('Weight Matix A')
#plt.axis([0, MODEL_NUM_DIM, 0, MODEL_ORDER])
#plt.axis('scaled')
plt.show()

plt.figure(figsize=(12, 6))
plt.imshow(B, cmap='gray', interpolation='nearest', aspect='auto')
plt.title('Weight Matix B')
#plt.axis([0, MODEL_NUM_HID, 0, MODEL_ORDER])
#plt.axis('scaled')
plt.show()

plt.figure(figsize=(12, 6))
plt.imshow([vbias], cmap='gray', interpolation='nearest', aspect='auto')
plt.title('Bias a')
#plt.axis([0, 1, 0, MODEL_NUM_DIM])
#plt.axis('scaled')
plt.show()

plt.figure(figsize=(12, 6))
plt.imshow([hbias], cmap='gray', interpolation='nearest', aspect='auto')
plt.title('Bias b')
#plt.axis([0, MODEL_NUM_HID, 0, 1])
#plt.axis('scaled')
plt.show()

"""

sess.close()
