# Paper Report 10: NeRF: Representing Scenes asNeural Radiance Fields for View Synthesis

**Paper** [https://arxiv.org/pdf/2003.08934](https://arxiv.org/pdf/2003.08934)

## Introduction

## Neural Radiance Field Scene Representation

As stated each scene is represented by an individual neural network.
Specifically the network encodes a function $F_{\Theta}$ that takes in a 3D point $\mathbf{x}$ as location and a 2D viewing direction $\mathbf{d} =(\theta, \phi) $ and
outputs the density of said point $\sigma$ as well as the color when viewed from the given direction $c=(R,G,B)$.

To explicitly force the predicted density to be direction independent (multiview consistency), an 8-layer MLP first only takes in $\mathbf{x}$ outputting both $\sigma$ and a
256-dimensional feature vector. The color $c$ is then produced by passing concatenating $d$ and the feature vector and passing this through a last fully connected layer.


## Volume Rendering with Neural Radiance Fields

A ray $\mathbf{r}(t)=\mathbf{o} + t\mathbf{d}$ with $t\in[t_n, t_f]$ is assigned the expected color 

$$ C(\mathbf{r})=\int_{t_n}^{t_f} dt\sigma(\mathbf{r}(t))\mathbf{c}(t, \mathbf{d}) \exp\left(-\int_{t_n}^t ds \sigma(\mathbf{r}(s)) \right). $$

The integral is approximated using probabilistic quadrature.
