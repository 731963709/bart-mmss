# BART-MMSS
[![](https://img.shields.io/badge/python-3.6+-blue.svg)](https://www.python.org/downloads/) 

[[Paper]](https://doi.org/10.1145/3539618.3591633) accepted at the SIGIR 2023:

**Adapting Generative Pretrained Language Model for Open-domain Multimodal Sentence Summarization**, by Dengtian Lin, [Liqiang Jing](https://liqiangjing.github.io/), [Xuemeng Song](https://xuemengsong.github.io/), [Meng Liu](https://mengliu1991.github.io/), [Teng Sun](https://teng-sun.github.io/), [Liqiang Nie](https://liqiangnie.github.io/).

## Abstract
Multimodal sentence summarization, aiming to generate a brief summary of the source sentence and image, is a new yet challenging task. Although existing methods have achieved compelling success, they still suffer from two key limitations: 1) lacking the adaptation of generative pre-trained language models for open-domain MMSS, and 2) lacking the explicit critical information modeling. To address these limitations, we propose a BART-MMSS framework, where BART is adopted as the backbone. To be specific, we propose a prompt-guided image encoding module to extract the source image feature. It leverages several soft to-be-learned prompts for image patch embedding, which facilitates the visual content injection to BART for open-domain MMSS tasks. Thereafter, we devise an explicit source critical token learning module to directly capture the critical tokens of the source sentence with the reference of the source image, where we incorporate explicit supervision to improve performance. Extensive experiments on a public dataset fully validate the superiority of our proposed method. In addition, the predicted tokens by the vision-guided key-token highlighting module can be easily understood by humans and hence improve the interpretability of our model.

## Model
<p align="center">
  <img align="center" src="Figure/prompt model figure.png" />
</p>
<p align="center">
  <b>Figure 1:</b> The proposed scheme based on BART backbone, which consists of four vital modules: Source Sentence Encoding, Prompt-guided Source Image Encoding, Explicit Critical Token Learning, and Multimodal Summary Generation.
</p>

## Data
we chose the [Multimodal Sentence Summarization(MMSS)](https://github.com/ZNLP/ZNLP-Dataset) dataset, which has been widely used to evaluate the performance of multimodal summarization models. The MMSS dataset contains $66,000$ samples, including $62,000$ for training, $2,000$ samples for validation, and $2,000$ for testing.


# Citations
if you find bart-mmss models helpful, feel free to cite the following publication:
```
@inproceedings{bart-mmss,
  author       = {Dengtian Lin and
                  Liqiang Jing and
                  Xuemeng Song and
                  Meng Liu and
                  Teng Sun and
                  Liqiang Nie},
  title        = {Adapting Generative Pretrained Language Model for Open-domain Multimodal
                  Sentence Summarization},
  booktitle    = {Proceedings of the 46th International {ACM} {SIGIR} Conference on
                  Research and Development in Information Retrieval, {SIGIR} 2023, Taipei,
                  Taiwan, July 23-27, 2023},
  pages        = {195--204},
  publisher    = {{ACM}},
  year         = {2023},
}
```
