
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
which makes the encoder a lot more efficient (reduces the input size). Experiments show that this helps with performance and shortens training time by more than a factor of 3.

The output of the ViT is a an equal number of encoded tokens that is now recombined with learned mask tokens in such a way that the overall list of tokens corresponds to
the inital (flattened) masked image, positional encoding is added once more. As the decoder is only needed for pretraining, it can be chosen flexibly and should be very
lightweight to reduce training time (the authors use a ViT with <10% compute of the encoder).
The decoder output is compared via pixelwise MSE with the true image. The authors also employ patchwise normalization before applying MSE, which improved reconstruction.

Note that one can efficiently perform the patch selection by simply shuffling the list of patches and keeping up to the number of needed ones. After the encoder mask encodings are added
and the list is unshuffled.


## Experiments

For their experiment, the authors pretrain the MAE setup with the ViT-Large architecture (from the original paper) as the encoder on the ImageNet-1k training set.
They try supervised linear probing, end-to-end finetung and some in between approaches all leading to good performances, showing that the prelearned features are in fact good.
One of the fully finetuned versions even achieves outperformance (at the time) on the highly competetive ImageNet-1k benchmark (87,8% accuracy), over fully supervised methods
using very advanced ViT setups (with vanilla ViT in the MAE).

Detailed ablation studies are performed. We will only cover two interesting points from this section. With respect to the masking ratio, they show that low and very high masking ratios
lead to lower performance. If only small sections of the image are missing it can be recovered via interpolation while recovery becomes mostly unfeasible for masking rates over 90%.
The performance peaks at around 75%, where the task is sufficiently hard while still being possible.

They also compare random masking to the blockwise masking, where large blocks of the image are removed and gridwise masking. The former makes the problem very hard leading to blurry reconstructions
(interestingly here the performance peaks at lower masking rates to balance the difficulty of the task) while the ladder makes it very easy. Ultimately these two alternatives perform worse than random sampling.

The authors additionally perform experiments with transfer learning on classification, detection/segmentation and semantic segmentation. The MAE is on par or outperforms SOTA (at the time) in all these tasks.
Again note that MAE uses vanilla ViT, while SOTA approaches often used highly optimized advanced architectures.

## Conclusion

Going from mostly supervised to mostly self-supervised training was one of the main enablers of scaling up language models. This paper shows with great clarity
how some simple adjustments can introduce this same success story into vision modelling. The use of MAE in the development of foundation models like METAs Segment Anything
(which I might cover soon) show that this is a really powerful tool in vision.

