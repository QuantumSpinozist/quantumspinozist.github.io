# Paper Report 6: An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale

**Paper**: [https://arxiv.org/pdf/2010.11929](https://arxiv.org/pdf/2010.11929)

## Introduction

After their invention in 2017, transformers quickly overtook all other model architectures in NLP. While there were experiments
trying to bring transformers into vision, there was a significant time span in which it was not clear if transformers
would remain as sequence to sequence machines, primarily staying in the NLP domain. This paper marked the turning point in this
question, as it showed that pure transformers in the form of vision transformers could outperform CNNs in vision tasks.

## Methodology

The input image $$ \mathbf{x} \in \mathbb{R}^{H\times W\times C}$$ ($$ H$$ is height, $$ W$$ is width, $$ C$$ is the number of colors) is
divided into $$N$$ patches which are then flattened into vectors such that we get an overall patch matrix $$  \mathbf{x}_p \in \mathbb{R}^{N \times (P^2 C)} $$ ($$ P$$ is the patch size).
We use a latent vector of constant dimension $$D$$ throughout the transformer, so we apply a linear projection $$\mathbf{E} \in \mathbb{R}^{(P^2 C) \times D}$$ to each
patch vector $$ \mathbf{x}_p^i $$.


In front of the $$N$$ patch vectors we 
add a class token vector $$\mathbf{x}_{\text{class}}$$ that is transformed into a classification vector at the end of the transformer. The token $$\mathbf{x}_{\text{class}}$$ is
learned during training and is independent of the input.
Like for text one also adds a positional embedding vector to every patch vector in the form of the matrix $$  \mathbf{E}_{\text{pos}} \in \mathbb{R}^{(N+1) \times D} $$
($$N+1$$ stems from the inclusion of the class token we just discussed). 
This encodes the position of the patch within the sequence and helps retain the spatial information
that was lost from splitting up the image (the authors report that using 2D positional encoding did not change the performance). 

The used transformer encoder is sketched in the image on the preview page, it mainly consists of a multi-head self attention and an MLP layer with layer normalization in between.
The full pipeline can be expressed as 

$$z_0 = \left[ x_{\text{class}}; x_p^1 \mathbf{E}; x_p^2 \mathbf{E}; \cdots ; x_p^N \mathbf{E} \right] + \mathbf{E}_{\text{pos}}, $$

$$z_\ell' = \text{MSA}(\text{LN}(z_{\ell-1})) + z_{\ell-1}, \quad \ell = 1 \dots L, $$

$$z_\ell = \text{MLP}(\text{LN}(z_\ell')) + z_\ell', \quad \ell = 1 \dots L, $$

$$ y = \text{LN}(z_0^L).$$

$$y$$ is the classification vector resulting from the class token as we discussed earlier. In the final step a clssification head is applied to $$y$$.
In pre-training it is an MLP, in finetuning it is simply a linear classifier. Fine tuning is preferable done on higher resolution images (compared to pre-training),
which leads to a larger number of patch vectors. If the sequence gets to long one needs to interpolate the pre-trained positional embedding to retain the
information about each patches location in the image correctly.

## Evaluation

The authors compare the ViT with ResNets and a CNN ViT hybrid model. As pre-training datasets they use ImageNet-1k, ImageNet-21k and JFT with 18k classes.
The trained models are then evaluated on differend downstream benchmark tasks namely, Imagenet-1k original validation labels (and cleaned up labels), CIFAR-10/100,
 Oxford-IIIT Pets and Oxford Flowers-102.

The ViTs generally outperform the other methods or are on par. I will not cover the conducted experiments in more detail as the great performance of ViTs on many vision tasks has been shown
time and time again since the release of this paper.

## Conclusion

This paper shows, in convincing fashion, something we have seen in many domains ever since. Namely that large amounts of data and scale
can (!) trump inductive bias, even in vision. At the same time we should not be to general and forget all subtelties.
While in NLP transformers are comfortably uncontested, there is much more of a coexistence in vision.
For example: in many real time applications, like autonomous driving, CNNs are still used more, since they are (at least believed to be by many)
faster at inference, especially on higher resolution images (this idea is very common but should be taken with a significant amount of skepticism,
as it is very contested, here is an article by Luca Beyer, who is among the first authors of the paper, on the matter [http://lucasb.eyer.be/articles/vit_cnn_speed.html](http://lucasb.eyer.be/articles/vit_cnn_speed.html)).
