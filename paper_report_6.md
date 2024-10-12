# Paper Report 6: An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale

**Paper**: [https://arxiv.org/pdf/2010.11929](https://arxiv.org/pdf/2010.11929)

## Introduction


## Methodology

The input image $$ \mathbf{x} \in \mathbb{R}^{H\cross W\cross C}$$ ($$ H$$ is height, $$ W$$ is width, $$ C$$ is the number of colors) is
divided into $$N$$ patches which are then flattened into vectors such that we get an overall patch matrix $$  \mathbf{x}_p \in \mathbb{R}^{N \cross (P^2 C)} $$ ($$ P$$ is the patch size).
We use a latent vector of constant dimension $$D$$ throughout the transformer, so we apply a linear projection $$\mathbf{E} \in \mathbb{R}^{(P^2 C) \cross D}$$ to each
patch vector $$ \mathbf{x}_p^i $$.

Like for text one adds a positional embedding vector to every patch vector. This encodes the position of the patch within the sequence and helps retain the spacial information
that was lost from splitting up the image (the authors report that using 2D positional encoding did not change the performance).
