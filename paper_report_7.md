
# Paper Report 7: Masked Autoencoders Are Scalable Vision Learners

**Paper**: [https://arxiv.org/pdf/2111.06377](https://arxiv.org/pdf/2111.06377)

## Introduction

Self-supervised pretraining has been key in the revolution of NLP we have seen in the last years. Both autoregressive models like GPT and masked autencoding models like BERT
are pretrained by making the model learn how to recover purposefully removed words. It seems plausible that this principle can be extended to vision, and indeed there
has been a lot of research in that direction, even before this publication (the paper extensively covers how vision and language differ in this respect, we will mostly omit
this discussion). This paper presents a simple and highly effective way of learning visual representations under self-supervision. The authors present a novel masked autoencoder (MAE) to achieve this.

## Methodology

Like in the original ViT paper, the image is cut into a number of image patches, from which a random selection is sampled to be masked. The mask-ratio is chosen purposefully high at 75% to force
the model to go beyond (non semantic) interpolation. The patches are fed into a ViT encoder with added positional encoding. Crucially no tokens for the masked patches are included at this stage,
which makes the encoder a lot more efficient (reduces the input size). 

The output of the ViT is a an equal number of encoded tokens that is now recombined with learned mask tokens in such a way that the overall list of tokens corresponds to
the inital (flattened) masked image, positional encoding is added once more. As the decoder is only needed for pretraining, it can be chosen flexibly and should be very
lightweight to reduce training time (the authors use one with <10% compute of the encoder).
The decoder output is compared via pixelwise MSE with the true image. The authors also employ patchwise normalization before applying MSE, which improved reconstruction.

Note that one can efficiently perform the patch selection by simply shuffling the list of patches and keeping up to the number of needed ones. After the encoder mask encodings are added
and the list is unshuffled.


## Experiments
