
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/learning-customized-visual-models-with/semi-supervised-image-classification-on-1)](https://paperswithcode.com/sota/semi-supervised-image-classification-on-1?p=learning-customized-visual-models-with)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/learning-customized-visual-models-with/semi-supervised-image-classification-on-2)](https://paperswithcode.com/sota/semi-supervised-image-classification-on-2?p=learning-customized-visual-models-with)

## REACT: Learning Customized Visual Models with Retrieval-Augmented Knowledge (CVPR 2023, Highlight 2.5%)

[Haotian Liu](https://hliu.cc), [Kilho Son](#), [Jianwei Yang](https://jwyang.github.io/), [Ce Liu](#), [Jianfeng Gao](https://www.microsoft.com/en-us/research/people/jfgao/), [Yong Jae Lee*](https://pages.cs.wisc.edu/~yongjaelee/), [Chunyuan Li*](https://chunyuan.li/)

[[Project Page](https://react-vl.github.io/)] [[Paper](https://arxiv.org/abs/2301.07094)]

![Teaser figure](figures/concept.gif)

- Introducing a customization stage to the lifecycle of foundation models!
- REACT customizes foundation models to downstream tasks without the need of any labeled data.

## :fire: News

* **[2023.03.29]** Code base and checkpoints are released.
* **[2023.03.25]** Our research paper is selected as <b>highlight</b> (2.5% acceptance rate)!
* **[2023.03.24]** Our new checkpoint based on OpenCLIP-G/14 achieves <b>81.0%</b> zero-shot on ImageNet, the <b>new SOTA</b> among public checkpoints!
* **[2023.02.28]** Paper is accepted to CVPR 2023.
* **[2023.01.17]** REACT paper is released.

## Code

### [:globe_with_meridians:	Stage 1: Retrieval](./react_retrieval)
REACT provides a pipeline that supports building index on a large dataset, and efficiently queries and retrieves relevant data for downstream tasks with information as simple as class names. See [`react_retrieval`](./react_retrieval) for details.

You may skip this step if you want to focus on building customized models on standard benchmarks like ImageNet-1K and ELEVATER, by directly using our retrieved indices.

### [:art: Stage 2: Customization](./react_customization) 

REACT proposes the efficient and effective *locked-text gated-image tuning* for tuning customized model on the retrieved dataset, with a performance improvement of up to 5.4% improvements on ImageNet. See [`react_customization`](./react_customization) for details.

## Pretrained Models

### ImageNet-1K

|                        | Baseline | REACT <br/> (Locked-Text) <br/> LAION-400M                                                                                                                                                                                                                            | REACT <br/> (Gated-Image) <br/> LAION-400M                                                                                                                                                                                                                             | REACT  <br/> (Gated-Image) <br/> LAION-2B                                                                                                        |
|------------------------|------|-----------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------|
| CLIP (B32, WIT-400M)   | 63.2 | 66.9 ([hf](https://huggingface.co/react-vl/react-in1k/blob/main/clip-vit-base-32-locked-text.pt)) | 68.6 ([hf](https://huggingface.co/react-vl/react-in1k/blob/main/clip-vit-base-32-gated-image.pt))                         | --                                                                                                                      |
| OpenCLIP (B32, L-400M) | 62.9 | 65.7 ([hf](https://huggingface.co/react-vl/react-in1k/blob/main/openclip-vit-base-32-locked-text.pt)) | 66.4 ([hf](https://huggingface.co/react-vl/react-in1k/blob/main/openclip-vit-base-32-gated-image.pt))                 | --                                                                                                                      |
| OpenCLIP (B32, L-2B)   | 66.6 | 67.5 ([hf](https://huggingface.co/react-vl/react-in1k/blob/main/openclip-laion2b-vit-base-32-locked-text.pt)) | 69.5 ([hf](https://huggingface.co/react-vl/react-in1k/blob/main/openclip-laion2b-vit-base-32-gated-image.pt)) | --                                                                                                                      |
| CLIP (B16, WIT-400M)   | 68.6 | 71.6 ([hf](https://huggingface.co/react-vl/react-in1k/blob/main/clip-vit-base-16-locked-text.pt)) | 73.4 ([hf](https://huggingface.co/react-vl/react-in1k/blob/main/clip-vit-base-16-gated-image.pt))                         | --                                                                                                                      |
| CLIP (L14, WIT-400M)   | 75.3 | -- | 78.1 ([hf](https://huggingface.co/react-vl/react-in1k/blob/main/clip-vit-large-14-gated-image.pt))                                                                                                                                    | 79.8 ([hf](https://huggingface.co/react-vl/react-in1k/blob/main/clip-vit-large-14-gated-image-laion2b.pt))     |
| OpenCLIP (L14, L-2B)   | 75.3 | -- | 76.4 ([hf](https://huggingface.co/react-vl/react-in1k/blob/main/openclip-vit-large-14-gated-image.pt))                                                                                                                                | 78.6 ([hf](https://huggingface.co/react-vl/react-in1k/blob/main/openclip-vit-large-14-gated-image-laion2b.pt)) |
| OpenCLIP (G14, L-2B)   | 80.1 | -- | --                                                                                                                                                                                                                                             | 81.0 ([hf](https://huggingface.co/react-vl/react-in1k/blob/main/openclip-vit-bigG-14-gated-image-laion2b.pt))  |

## Citation
```
@article{liu2023react,
  author      = {Liu, Haotian and Son, Kilho and Yang, Jianwei and Liu, Ce and Gao, Jianfeng and Lee, Yong Jae and Li, Chunyuan},
  title       = {Learning Customized Visual Models with Retrieval-Augmented Knowledge},
  publisher   = {CVPR},
  year        = {2023},
}
```

## Acknowledgement

We are grateful for the contributions of several open-source projects, including [CLIP](https://github.com/openai/CLIP), [OpenCLIP](https://github.com/mlfoundations/open_clip), [LAION.AI](https://laion.ai/), [FAISS](https://github.com/facebookresearch/faiss), [Autofaiss](https://github.com/criteo/autofaiss), [img2dataset](https://github.com/rom1504/img2dataset), and [ELEVATER](https://github.com/Computer-Vision-in-the-Wild/Elevater_Toolkit_IC).

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
