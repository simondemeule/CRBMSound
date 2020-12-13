# Introduction

This project is an exploration of the use of conditional restricted Boltzmann machines for sound synthesis in the context of music. This work was guided by professor Denis Pankratov of Concordia University, and was possible thanks to an NSERC undergraduate research award. This research was also facilitated by the [xRMB library](https://github.com/omimo/xRBM), which served as a great starting point for our experiments.

Most of this project focusses on the use of a CRBM in a feedback loop operating on audio samples; the model is trained to predict a single audio sample by conditioning on a few previous samples. During generation, the model infers a sample, and this prediction is added to the past context of the model to generate new samples recursively.

# Details

In this simple form, the algorithm closely ressembles a linear system with added noise terms from sampling. A formal proof for this was attempted, however the long-term impact of sampling noise has proven hard to work with. In practice, however, the Z-transform and pole-representation of linear systems can be used to conceptualize and transform model parameters; this way it is possible to take a set of leant parameters and transform them so that the generator produces sounds with altered pitch, resonance, and amplitude. CRBMTimeInvariant implements methods that allow conversion between the model parameters, linear system polynomial and pole representation, along with functions that allow rotation, scaling, and displacement of poles; manipulations that have a meaningful impact on the generated sound. This representation also gives a necessary but not sufficient condition for model stability, borrowed from linear systems: all poles must lie within the unit circle. Again, the insufficient nature of this guarantee comes from the cumulative sampling noise of the model.

A simple modification which was explored in order to limit model instability is the addition of soft clipping, a technique often used in digital signal processing techniques for music whenever feedback loops are used (reverbs, echoes, flangers, etc). We designed a softclip function that is purely linear in the signal's central range, and that smoothly tapers off at its extremes; the rationale being that it would not impart nonlinearities on the process in its usual operating range. We found that obviously prevented exponentially large output from being generated, however instabilities and poor behaviour remained. This did not address the source of the issue, but rather bounded its explosive impact. 

Another modification of this model is explored; it is coined "multiscale sampling". Rather than conditioning on a simple contiguous sequence of past samples, the model can condition on samples averaged over bins that get exponentially larger as they reach further into the past. With an exponent of two, this representation is equivalent to a (nonsingular) linear transform of a subset of the coefficients of the Haar wavelet transform, where at every octave only the information of the temporally closest wavelet is kept. This offers a way of conceptualizing the sampling scheme as a way of compromising between the capture of time and frequency information; by using this scheme, the model retains its knowledge of high-frequency information in the immediate past, and also reaches further back to capture more low-frequency content, something that the simple sampling scheme cannot do. It would be possible to explore this idea with other wavelets, however the Haar wavelet is very attractive since it is mathematically simple; given the averaged bins calculated at the last time step, the new coefficients differ only by the respective samples that enter or leave the bin; this means we can obtain the new coefficients in linear time relative to the number of bins, all while conditioning over an exponential number of samples. This is also implemented by CRBMTimeInvariant; however, this implies an exponentially large number of linear system coefficients that are constrained together, which makes it difficult to manipulate the polynomial and pole representation of the model. This modification seemed not to have a significative impact on the capacity of the model however.

# Conclusion

The main conclusion of these experiments is that using CRBMs in a simple sample-level feedback system yields a parametrization that can be interpreted and manipulated in a very interesting and convenient way, which unfortunately this comes at a heavy cost: the implication is that CRBMs used in this fashion are not meaningfully more expressive than linear systems. They are mostly successful at learning simple percussive sounds that decay exponentially, but struggle with more complex sounds. This may explain why the alternate sampling scheme yielded no obvious improvements. After arriving at this conclusion, our attention shifted to using CRBMs with multivariate data derived by the FFT, so that more complex relationships could be captured. We were unfortunately not able to complete this implementation. This project was an incredible learning experience in that it allowed me to delve into the world of machine learning; it allowed me to survey the theory of probabilistic graphical models, energy based models and generative models, and to gain experience in implementing these methods practically.

