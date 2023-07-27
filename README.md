This repository contains code for adapting the X-MOD model to Xhosa, Yoruba, and Arabic under a low-resource domain.
There is both a huggingface implementation located in [code](code) and fairseq implementation in [swissbert](swissbert) based on the original [SwissBERT repo](https://github.com/ZurichNLP/swissbert).
Work is based on the [X-MOD](https://arxiv.org/abs/2205.06266) and [SwissBERT](https://arxiv.org/abs/2303.13310) papers.

[AfroMAFT corpus](https://zenodo.org/record/6990611) is used for continued pretraining and [MasakhaNER 2.0 dataset](https://github.com/masakhane-io/masakhane-ner/tree/main/MasakhaNER2.0) is used for downstream NER finetuning.
