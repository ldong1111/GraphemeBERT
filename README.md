# GraphemeBERT
This is the source code of the paper "Neural Grapheme-to-Phoneme Conversion with Pretrained Grapheme Models".

This paper is accepted by ICASSP 2022.

## Overview
In this paper, we proposes a pre-trained grapheme model called grapheme BERT (GBERT), which is built by self-supervised training on a large, language-specific word list with only grapheme information. We borrowed the mask machanism of BERT to capture the contextual grapheme information in a word. Furthermore, two approaches are developed to incorporate GBERT into the state-of-the-art Transformer-based G2P model, i.e., fine-tuning GBERT or fusing GBERT into the Transformer model by attention. Experimental results on the Dutch, Serbo-Croatian, Bulgarian and Korean datasets of the SIGMORPHON 2021 G2P task confirm the effectiveness of our GBERT-based G2P models under both medium-resource and low-resource data conditions.

![GBERT](https://github.com/ldong1111/GraphemeBERT/blob/main/GBERT.png)

### Some Reference codes
[Transformer](https://github.com/bentrevett/pytorch-seq2seq)

[BERT](https://github.com/codertimo/BERT-pytorch)

[BERT-fused model (GBERT attention)](https://github.com/bert-nmt/bert-nmt)

## Todo
The source code will come soon.
