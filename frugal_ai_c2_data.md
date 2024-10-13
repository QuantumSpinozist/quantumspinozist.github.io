# Frugal AI - Data Frugal AI

## Motivation

## Preprocessing with Frugal Data

### Data Augmentation

When little data is available, good data augmentation often becomes crucial. In the context of
images common techniques include flipping, changing the colors, saturation or brightness, 
cropping and rotating the image. In specialized domains like medical imaging there are also more
specialized domain-oriented augmentations. Their application often requires more time investment
and some amount of domain knowledge.

Generic augmentations for text are synonym replacement, random insertion, random swapping and random deletion.
Again for specialized texts domain-specific augmentations might be applicable.

A more sophisticated technique for augmentation is sample mixing, where one combines two samples. A popular implementation
of this approach is MixUp. In MixUp one creates a convex combination of two existing samples $$ \lambda x_a + (1-\lambda)x_b $$
(the same for the labels) with a beta sampled parameter $$\lambda$$.

When using data augmentation it is important to not augment validation data as this changes the target distribution.
One should also not use too much augmentation and generally look at what has worked for others. Like for most things in DL,
a lot of experimentation is required.

### Data Generation



## Zero- , One- and Few-Shot Learning

### Zero-Shot Learning

### One-Shot Learning

### Few-Shot Learning

## Imbalanced Data

## Streamed Data - Continual Learning
