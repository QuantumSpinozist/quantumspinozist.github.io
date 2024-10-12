# Paper Report 6: An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale

**Paper**: [https://arxiv.org/pdf/2010.11929](https://arxiv.org/pdf/2010.11929)

## Introduction


## Methodology

The input image $$ \mathbf{x} \in \mathbb{R}^{H\times W\times C}$$ ($$ H$$ is height, $$ W$$ is width, $$ C$$ is the number of colors) is
divided into $$N$$ patches which are then flattened into vectors such that we get an overall patch matrix $$  \mathbf{x}_p \in \mathbb{R}^{N \times (P^2 C)} $$ ($$ P$$ is the patch size).
We use a latent vector of constant dimension $$D$$ throughout the transformer, so we apply a linear projection $$\mathbf{E} \in \mathbb{R}^{(P^2 C) \times D}$$ to each
patch vector $$ \mathbf{x}_p^i $$.

Like for text one adds a positional embedding vector to every patch vector. This encodes the position of the patch within the sequence and helps retain the spacial information
that was lost from splitting up the image (the authors report that using 2D positional encoding did not change the performance). In front of the $$N$$ patch vectors we also 
add a class token vector $$\mathbf{x}_{\text{class}}$$ that is transformed into a classification vector at the end of the transformer. The token $$\mathbf{x}_{\text{class}}$$ is
learned during training and is independent of the input.

The used transformer encoder is sketched in the image on the preview page, it mainly consists of a self attention and an MLP layer with layer normalization in between.
The full pipeline can be expressed as 

$$z_0 &= \left[ x_{\text{class}}; x_p^1 \mathbf{E}; x_p^2 \mathbf{E}; \cdots ; x_p^N \mathbf{E} \right] + \mathbf{E}_{\text{pos}}, $$

$$\mathbf{E} &\in \mathbb{R}^{(P^2 \cdot C) \times D}, \quad \mathbf{E}_{\text{pos}} \in \mathbb{R}^{(N+1) \times D} $$

$$z_\ell' &= \text{MSA}(\text{LN}(z_{\ell-1})) + z_{\ell-1}, \quad \ell = 1 \dots L, $$

$$z_\ell &= \text{MLP}(\text{LN}(z_\ell')) + z_\ell', \quad \ell = 1 \dots L, $$

$$ y &= \text{LN}(z_0^L).$$

