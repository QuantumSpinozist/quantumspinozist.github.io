# Book Report: The Principles of Deep Learning Theory - Part 1, Neural Networks at Initialization

**Book** [https://arxiv.org/pdf/2106.10165](https://arxiv.org/pdf/2106.10165)

## -1. Introduction to this Report
This report aims to summarize the main takeaways of "The Principles of Deep Learning Theory" in a more or less easily consumable
manor. It does not in any way substitute the original text, especially since I will radically skip the majority of the technical derivations, 
which one might rightfully argue really form the meat of the book (in particular to someone who wishes to adapt some of the techniques for
their own theoretical undertakings). I highly suggest reading the book, even if the long messy calculations
can be somewhat hard to get through, especially for people who are unfamiliar with field theory. Under this approach
this report should be seen as a mostly supplementary ressource, designed to help you see both the forest and the trees.
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

Concretely the authors goal is to analytically calculate the leading orders to the probability distribution of the MLPs function

$$ p(f^*) = p(f(x; \theta^*) | \text{learning algorithm}; \text{training data}).$$

The randomness explicitly comes in by the initialization of $$ \theta $$ being randomly sampled.
Now MLPs really only have two relevant scaling hyperparameters, namely the layer width $$n$$ and the
nets depth $$L$$. Looking at systems with an infinite number of degrees of freedom often leads to simplifications in physics, we
observe something similar here. While the situation of infinite depth is hopeless, the infinitely wide network is
a well studied limit, that greatly simplifies the problem. Here our network collapses into a linear operator and the distribution
from above becomes fully gaussian (we will see this later). Since this makes representation learning impossible a weaker limit is chosen.
Specifically we wish to obtain the linear order to

$$ p(f^*) = p^{(0)}(f^*) + p^{(1)}(f^*) \frac{1}{n} + \mathcal{O}(\frac{1}{n^2})$$

(this is the leading order we did not specify earlier). This corresponds to the limit of large but finite
width network, that is in general quite realistic to actual practice. We can also characterize the network using the relative quantity

$$ r = \frac{L}{n} .$$

For $$ r = 0$$ we again have the infinte width limit. For $$0<r\ll 1$$ we get an effectively deep network, the width expansion truncates and is analytically
tractable. For $$r\gg 1$$ the network is overly deep and becomes entirely chaotic. In this regime inter layer fluctuations dominate and our expansion becomes intractable.

So in the following we will analytically study large but finite width MLPs to develop an effective theory for the behaviour of non trivial neural networks.

## 1. Preparation
This chapter covers some of the recurring methods used in the coming calculations. Since I will not cover them in detail I will go through most of this section rather quickly.
The authors start out with a discussion of gaussian integrals and show how moments of multidimensional gaussians can be evaluated using Wicks theorem from QFT (also known as Isserlis theorem in
probability theory). They introduce the connected two point and four point (also six point etc) correlator as the simplest possible observables for gaussians. Nearly-Gaussian distributions
now get a first definition as those for which all connected correlators after the two point one are small.

Next the action $$ S(z) $$ (sometimes negative log probability in probability theory) of a distribution is introduced as any function that satisifes

$$ p(z) \propto \exp(-S(z)) $$

which fixes it up to an additive constant. Now for some small $$\epsilon$$ the simplest possible nearly gaussian distribution has the action

$$ S(z) = \frac{1}{2} K^{\mu \nu} z_{\mu} z_{\nu} + \frac{\epsilon}{4!} V^{\mu \nu \rho \lambda} z_{\mu} z_{\nu} z_{\rho} z_{\lambda}.$$

Unlike the book we use einstein notation, meaning we sum over the double indices. For $$\epsilon$$ being zero we recover the purely quadratic expression of the gaussian.
The authors explicitly demonstrate that the connected four point correlator is proportional to the four point coupling tensor $$ \epsilon V $$, which connects this to our earlier definition.

## 2. Neural Networks
Most of this chapter covers the basic ingredients needed for neural networks, which we will skip. For the initialization of the parameters at layer $$ l $$
we introduce the notation

$$ \mathbb{E} [b_{i_1}^{(l)} b_{i_2}^{(l)}] = \delta_{i_1 i_2} C_b^{(l)} $$

$$ \mathbb{E} [W_{i_1 j_1}^{(l)} W_{i_2 j_2}^{(l)}] =  \delta_{i_1 i_2}  \delta_{j_1 j_2} \frac{C_W^{(l)}}{n_l -1}.$$

As is the norm we assume they are normally distributed. So studying the MLP at initialization means we need to evaluate the integral

$$ p\left( z^{(L)} \middle| \mathcal{D} \right) = \int \left[ \prod_{\mu=1}^{P} d\theta_\mu \right] p\left( z^{(L)} \middle| \theta, \mathcal{D} \right) p(\theta)
 $$

 where $$ \mathcal{D} $$ denotes the given input data. Using a dirac delta distributions this can also be rewritten as
 
 $$ p\left( z^{\text{out}} \middle| \mathcal{D} \right) = \int \left[ \prod_{\mu=1}^{P} d\theta_\mu \right] p(\theta) 
 \left[ \prod_{i=1}^{n_{\text{out}}} \prod_{\alpha \in \mathcal{D}} \delta \left( z_{i;\alpha}^{\text{out}} - f_i(x_\alpha; \theta) \right) \right].
 $$

## 3. Effective Theory of Deep Linear Networks at Initialization

Before going to the actual MLP, the authors do the needed calculations for a toy model, namely the deep linear network (without biases).
In this simplified setting one can explicitly write the output of the $$ l $$-th layer as

$$ z_{i_\ell;\alpha}^{(\ell)} = \sum_{j_0=1}^{n_0} \sum_{j_1=1}^{n_1} \cdots \sum_{j_{\ell-1}=1}^{n_{\ell-1}} W_{i_\ell j_{\ell-1}}^{(\ell)} W_{j_{\ell-1} j_{\ell-2}}^{(\ell-1)} \cdots W_{j_1 j_0}^{(1)} x_{j_0;\alpha}.$$

Notice that one can fully contract the sum ultimately resulting in a single matrix multiplication (I assume most readers already knew this anyway).
According to the program we have layed out in the previous sections we now calculate the correlators. The single point correlator $$ \mathbb{E} [z_{i_\ell;\alpha}^{(\ell)}] $$
vanishes as one would expect. We can express the two point correlator as

$$ G^{(\ell)}_{\alpha_1 \alpha_2} = \left( C_W \right)^\ell G^{(0)}_{\alpha_1 \alpha_2},
 $$

where we have defined $$ G^{(\ell)}_{\alpha_1 \alpha_2} $$ as the $$ l $$-th layer correlator for both indices being the same and $$ G^{(0)}_{\alpha_1 \alpha_2} \equiv \frac{1}{n_0} \sum_{i=1}^{n_0} x_{i;\alpha_1} x_{i;\alpha_2}.$$ 
Notice that this iteration leads to a fixed point at infinity for $$ C_W > 1 $$, one at $$ 0 $$ for $$ C_W < 1 $$ and one at the input correlation for $$ C_W = 1 $$. In the language of the Renormalization Group (more on that later)
the first two fix points are called trivial, because the value zero or infinity is approached exponentially quickly. The last point is called non trivial since the fix point stabilizes at a finite value. We will later see 
qualitatively similar results for other activation functions.

 From now on we will denote self correlators (where all indices are the same) by the number of indices. We move on by calculating the four point self correlator

$$ G_4^{(l)} = \left[ \prod_{l' = 1}^{l-1} (1+\frac{1}{n_{l'}}) \right] (G_2^{(l)})^2 = C_W^{2l} \left[ \prod_{l' = 1}^{l-1} (1+\frac{1}{n_{l'}}) \right] (G_2^{(l)})^2.$$

We can see that for the infinite width limit the four point correlator is equal to the two point one. This is as expected since the distribution stays gaussian in this case and all higher correlators are given by the two point one.
Before we analyze this result a bit more we want to give the higher correlators namely

$$ G_{2m}^{(l)} =  \left[ \prod_{l' = 1}^{l-1} c_{2m}(n_{l'}) \right] C_W^{ml} (G_2^{(0)})^m$$

where we have introduced the general combinatorial factor

$$ c_{2m}(n) = \frac{(\frac{n}{2} - 1 + m)!}{(\frac{n}{2} - 1)!} \left(\frac{2}{n}\right)^m.$$

Again it is easy to see that this factor goes to one in the infinite width limit. On the other hand the factor is always $$ >1$$ for $$ m>1 $$ and finite width. This means that in the infinte depth limit the higher
correlators get exponentially large and the resulting behaviour is highly chaotic. Finally we can expand the connected correlators (so those where all lower correlations are substracted) in $$ r $$ giving us

$$ G_4^{(L)} - (G_2^{(L)})^2 = 2r (G_2^{(0)})^2 + \mathcal{O}(r^2)$$

$$ G_6^{(L)} - 3G_2^{(L)} G_4^{(L)} + 2(G_2^{(L)})^3 = 12r^2 (G_2^{(0)})^3 + \mathcal{O}(r^3).$$

So we see that for the realistic case of small $$r$$ we get a nearly gaussian distribution since the connected four point correlator is linearly suppresed in $$r$$ while
all higher connected correlators are suppressed in even higher powers if $$ r $$. This is a theme that will return again and again throughout our explorations. It enables
us to truncate the distribution after a given order of $$r$$.

## 4. RG Flow of Preactivations



## 5. Effective Theory of Preactivations at Initialization
