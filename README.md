# SVGFaith

SVGFaith is a two-layer meta-evaluation of faithfulness metrics for text-to-SVG generation.

[Paper](https://arxiv.org/abs/2611.06048) · [Decomposer](https://huggingface.co/Yukinonooo/SVGFaith-Decomposer)

Text-to-SVG systems are usually scored by applying a raster metric to a rendered SVG. SVGFaith tests whether that score actually prefers the more faithful candidate. SVGFaith-Diagnostic contains 1,842 controlled contrastive pairs across eight capability axes, split into 921 image-side and 921 text-side pairs. SVGFaith-Arena contains 1,327 human comparisons of outputs from six generators.

SVGFaith-Score decomposes a prompt into a dependency graph of atomic visual requirements, scores each requirement with a yes/no visual question, and discounts a requirement when its prerequisites score low. At a tie threshold of 0.01 it reaches 0.880 pairwise accuracy on Arena and 0.941 / 0.953 on the image and text sides of Diagnostic. Diagnostic numbers are unweighted means of the eight axes. Ties receive no credit.

## Repository layout

```text
data/diagnostic/     1,842 contrastive pairs, SVG and PNG
data/arena/          1,327 human pairs and their renderings
scores/              released SVGFaith-Score outputs
svgfaith/            aggregation and the paper's accuracy calculation
```

## Reproduce the headline numbers

```bash
python -m svgfaith.evaluate scores/arena.jsonl scores/diagnostic_image.jsonl scores/diagnostic_text.jsonl
```

Arena is reported as wins over all 1,327 pairs. Each Diagnostic direction is the mean of eight axis accuracies. The released Arena scores are 1,168 wins, 37 ties, and 122 losses.

```bash
python -m unittest tests.test_aggregate
```

## Metric

SVGFaith-Decomposer is a Qwen3.5-9B model trained with teacher distillation, rejection-sampling supervised fine-tuning, and GRPO. It reads only the prompt. The system prompt is [`svgfaith/prompts/decomposer_system.txt`](svgfaith/prompts/decomposer_system.txt). Weights: [`Yukinonooo/SVGFaith-Decomposer`](https://huggingface.co/Yukinonooo/SVGFaith-Decomposer).

Each requirement is then asked to the unmodified `Qwen/Qwen3.5-9B` with thinking disabled. The question is phrased so that Yes means the requirement is satisfied. An absence requirement asks whether the image is free of the forbidden content. The requirement probability is

```text
P(Yes) / (P(Yes) + P(No))
```

at the first answer token. Unique-ancestor aggregation with alpha 0.5 is implemented in `svgfaith/aggregate.py`. If an apple is present with probability 0.8 and "the apple is red" scores 0.9, the color requirement contributes `0.9 * (0.5 + 0.5 * 0.8) = 0.81`.

## Data

Diagnostic construction, the eight axes, and the human review funnel are in Section 2 and Appendix B of the paper. `data/diagnostic/manifest.jsonl` is the index: 1,799 pairs kept by all three annotators and 43 pairs added by expert review. Annotator identities and raw model traces are not included.

Arena labels follow a two-of-three majority. Pairs that remained ties, and 25 pairs removed in the published label review, are not in the 1,327-pair evaluation set. See `data/arena/README.md`.

## License

Code is Apache 2.0. Benchmark files and released scores are CC BY 4.0. See `LICENSE` and `DATA_LICENSE`.

## Citation

```bibtex
@misc{liang2026svgfaith,
  title = {SVGFaith: A Two-Layer Meta-Evaluation of Faithfulness Metrics for Text-to-SVG Generation},
  author = {Liang, Guotao and Zhou, Haitao and Liu, Biao and Liao, Huanxuan and Qiao, Lingfeng and Zhang, Qian-Wen and Yu, Qian and Yin, Di and Sun, Xing},
  year = {2026},
  eprint = {2611.06048},
  archivePrefix = {arXiv},
  primaryClass = {cs.CV},
  url = {https://arxiv.org/abs/2611.06048}
}
```

## 中文

这是论文 [arXiv:2611.06048](https://arxiv.org/abs/2611.06048) 的代码、评测数据和拆题模型发布。Diagnostic 1,842 对，Arena 1,327 对。拆题器在 [Hugging Face](https://huggingface.co/Yukinonooo/SVGFaith-Decomposer)。上面的命令会复现论文 Table 2 里 SVGFaith-Score 的三个主数字。
