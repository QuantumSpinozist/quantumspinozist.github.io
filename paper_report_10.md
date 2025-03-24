# Paper Report 10: NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis

**Paper** [https://arxiv.org/pdf/2003.08934](https://arxiv.org/pdf/2003.08934)

## Introduction

NeRF marked a big breakthrough for scene representation. It noticeably differs from most the usual Deep Learning approach where a network is trained to perform a task in general,
in that one trains a new network for each individual scene.

## Neural Radiance Field Scene Representation

As stated each scene is represented by an individual neural network.
Specifically the network encodes a function $F_{\Theta}$ that takes in a 3D point $\mathbf{x} = (x, y, z)$ as location and a 2D viewing direction $\mathbf{d} =(\theta, \phi) $ and
outputs the density of said point $\sigma$ as well as the color when viewed from the given direction $c=(R,G,B)$.

To explicitly force the predicted density to be direction independent (multiview consistency), an 8-layer MLP first only takes in $\mathbf{x}$ outputting both $\sigma$ and a
256-dimensional feature vector. The color $c$ is then produced by passing concatenating $d$ and the feature vector and passing this through a last fully connected layer.


## Volume Rendering with Neural Radiance Fields

A ray $\mathbf{r}(t)=\mathbf{o} + t\mathbf{d}$ with $t\in[t_n, t_f]$ is assigned the expected color 

$$ C(\mathbf{r})=\int_{t_n}^{t_f} dt\sigma(\mathbf{r}(t))\mathbf{c}(t, \mathbf{d}) \exp\left(-\int_{t_n}^t ds \sigma(\mathbf{r}(s)) \right). $$

The integral is approximated using probabilistic quadrature.

## Optimizing the NeRF 
A few additional tricks are needed to make this setup work well in practice.

First the authors use what they call positional encoding (similar to whats used in transformers but for a different purpose).
To make it easier for the MLP todeal with both low and high frequency variation the positional input variables $(x, y, z, \theta, \phi)$ (angles are converted to cartesian) are projected into a higher dimensional space
using

$$ \gamma(p) = \left( \sin(2^0 \pi p), \cos(2^0 \pi p), \cdots, \sin(2^{L-1} \pi p), \cos(2^{L-1} \pi p) \right). $$

They also optimize the rendering process using hierarchical volume sampling. To do this they have two networks: one "coarse" and one "fine". First they sample a set number of locations on the beam using
strtified sampling and evaluate the "coarse" network on them. Then important sections of the beam are samples using this information and evaluated using the other net.

## Experimental Results
The authors show that their method outperforms previous approaches both qualitatively and quantitatively (we do not cover this in further detail).
