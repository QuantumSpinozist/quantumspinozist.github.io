
# Paper Report 7: Attention Is All You Need

**Paper**: [https://arxiv.org/pdf/1706.03762](https://arxiv.org/pdf/1706.03762)

## Introduction


## Methodology

### Micro Architecture - Attention and other Components

The authors introduce the canonical form of attention today, namely scaled dot-product attention specifically as self attention.
For a query vector $$ \mathbf{Q} $$, a key vector $$\mathbf{K}$$ and a value vector $$\mathbf{V}$$ attention is defined as

$$ \text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left( \frac{\mathbf{Q} \mathbf{K}^T}{\sqrt{d_k}} \right) \mathbf{V} .$$

In practice these quantities will be matrices representing a sequence of vectors (representing one token each). The softmax part is the attention score
matrix where entry $$i, j $$ corresponds to how much attention the $$i$$-th query vector should give to the $$j$$-th key vector. The $$i$$-th attention vector
will then be the sum of all possible value vectors weighted by the corresponding attention scores.

Multi head attention now simply projects every $$d$$ dimensional query, key, value vector down using $$h$$ different linear layers, 
applies attention separately and concatenates the results back together. The resulting vector is passed through another linear layer to get
back $$d$$ dimensional vectors. In formulae this reads as


$$ \text{MultiHead}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) \mathbf{W}^O $$

where each head is computed as

$$ \text{head}_i = \text{Attention}(\mathbf{Q} \mathbf{W}_i^Q, \mathbf{K} \mathbf{W}_i^K, \mathbf{V} \mathbf{W}_i^V) .$$

The idea behind this is that each head learns a different attention mechanism, making the model more flexible.




### Macro Architecture - Encoder and Decoder




## Experiments
