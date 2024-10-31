# Paper Report 9: Siamese Neural Networks for One-shot Image Recognition

**Paper** [https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf)

## Introduction

I reimplemented this paper over on my github ([https://github.com/QuantumSpinozist/One-Shot-Image-Recognition](One-Shot-Image-Recognition)).

I personally find few and one shot learning really interesting. For many applications, having very few data points available is more the norm than
the opposite, so trying to get the most out of your small amount data is both theoretically and practically interesting. One shot learning pushes this to
a provisional limit (zero shot learning shall be somewhat excluded as it usually works quite differently). In a time with more and more powerful
foundation models in many domains, few short tasks become more and more feasible. This paper comes from before those developments, but it presents
a general blueprint for how to solve one shot tasks by learning a metric. This basic idea is still present in many recent models (like CLIP for example).

The authors train a Siamese Network to identify matching images of handwritten characters from the Omniglot dataset. They then use said network to solve classification
tasks with unseen alphabets (from Omniglot) where only one example per class is presented to the model, achieving very good accuracy.

## Method

The backbone of the Siamese Network is made up of two identical CNNs (refer to the paper for the specific architecture of the CNN). The euclidean distance between their output feature vectors is computed on top of which a
one layer classifier is placed. This model is presented with pairs of images and is tasked to output 1 if they belong to the same class and 0 if they do not.

One can use a simple BCE loss for this problem (remember one sample now corresponds to two images). The authors employ data augmentation of the training images and allow different learning rates
per layer. They also tune the hyperparameters of the model and the training via bayesian optimization, based on the one shot validation performance as the objective. We will cover the one shot tasks in detail
in the next section.


## Experiments

The model is trained on the Omniglot dataset. Omniglot contains images of hand written characters from 50 different alphabets. Every alphabet has some number of different characters, and each
character was drawn one time each by 20 different people. We train the model on the data from a subset of alphabets while reserving the rest for the one shot evaluation (also some for verification).

To create a one shot classification task we pick one of the alphabets the model has not seen and select 20 of the characters.
The model now gets access to one example each for every one of the 20 classes and has to classify the other samples from those 20 characters it has not seen.
Using the Siamese Network one can do this by simply picking the class whose example maximizes the similarity score (as modelled by the net) with the
sample.

The authors achieve a one shot accuracy of 92% over 400 tasks. The human benchmark they provide lies at 95.5% and the best model is only slightly worse at 95.2%.
The best model uses Hierarchical Bayesian Program Learning, which requires extensive prior knowledge about the drawing process.

