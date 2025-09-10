# Peri-LN-Experiment

This is a Deep Learning course (NTNU Laboratory Training Experiment) using PyTorch.

---

## Experiment Result
* Datasets is CUB 200 2011
* Training on ResNet50
* Comparison of different LayerNorm positions: **pre-LN, post-LN, and peri-LN**
* The peri-LN setting follows *Peri-LN: Revisiting Layer Normalization in the Transformer Architecture* ([Xu et al., 2023](https://arxiv.org/abs/2305.13305))
* `model.py` is adapted based on the official PyTorch [ResNet implementation](https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py)

<p float="left">
  <img src="result/numeric-result.png" alt="Experiment Result" width="700" />
</p>

<p float="left">
  <img src="result/train-result.png" alt="Training Result" width="700" />
</p>

```bibtex
@article{kim2025peri,
  title={Peri-LN: Revisiting Layer Normalization in the Transformer Architecture},
  author={Kim, Jeonghoon and Lee, Byeongchan and Park, Cheonbok and Oh, Yeontaek and Kim, Beomjun and Yoo, Taehwan and Shin, Seongjin and Han, Dongyoon and Shin, Jinwoo and Yoo, Kang Min},
  journal={arXiv e-prints},
  pages={arXiv--2502},
  year={2025}
}
