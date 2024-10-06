# Book Report: The Principles of Deep Learning Theory - Part 1, Neural Networks at Initialization

**Book** [https://arxiv.org/pdf/2106.10165](https://arxiv.org/pdf/2106.10165)

## -1. Introduction to this Report
This report aims to summarize the main takeaways of "The Principles of Deep Learning Theory" in a more or less easily consumable
manor. It does not in any way substitute the original text, especially since I will radically skip the majority of the technical derivations, 
which one might rightfully argue really form the meat of the book (in particular to someone who wishes to adapt some of the techniques for
their own theoretical undertakings). I highly suggest reading the book, even if the long messy calculations
can be somewhat hard to get through, especially for people who are unfamiliar with field theory. Under this approach
this report should be seen as a mostly supplementary ressource, designed to help you see both forest and the trees.
Nevertheless I hope this text can also be appreciated by people who do not intend to read the whole book.

Unlike in my usual reports I have stuck with the structure of the book in terms of chapter numbering, but I have slightly adjusted the chapter titles at times.
Part 1 will cover the first half of the book which studies MLPs at weight initialization. Unlike the book I will also assume that the reader is largely familiar
with the standard repertoire of deep learning.

## 0. Introduction
Even if the roots of Deep Learning go back all the way to the late 1950s, the research landscape we see today is only around 20 years old, even
by a conservative definition. In this short time Deep Learning has positioned itself as an extremely successful framework for artificial
intelligence, and radically changed the world already. Nevertheless there is a huge gap between this empirically observed success and 
our theoretical understanding of that success. The young research field of Deep Learning theory aims to close or at least decrease that gap.

The authors want to contribute to this endeavour by developing an effective theory of neural networks (specifically MLPs other architectures are not covered).
In this context effective theory is a technical term from theoretical physics. In physics an
effective theory describes a system at a chosen scale by averaging over the degrees of freedom (parameters) that occur at lower scales. Thermodynamics is an effective theory
describing systems in terms of macroscopic quantities like temperature and pressure, while statistical mechanics is the theory describing the underlying small scale degrees of
freedom. Curiously in the case of neural networks the situation is figuratively upside down when compared to physics. In physics one usually starts of with an effective theory
and wishes to discover the smaller degrees of freedom (think of our example). But for deep learning the microscopic theory describing the interactions between neurons is
the starting point and also the thing that is actually implemented in the natural habitat of neural nets, namely PyTorch and Tensorflow. The authors now wish to go from
this true theory to an effective one to gain some theoretical insights about how neural networks work.



## 1. Preparation

## 2. Neural Networks

## 3. Effective Theory of Deep Linear Networks at Initialization

## 4. RG Flow of Preactivations

## 5. Effective Theory of Preactivations at Initialization
