# Youtu-SVGFaith Diagnostic Set

The Diagnostic Set contains 1,842 controlled contrastive pairs across eight capabilities: object identity, part completeness, spatial relation, dense long prompt, counting, negation, comparative size, and attribute binding.

- Image side (`direction=image`): a text prompt with two SVG/PNG candidates.
- Text side (`direction=text`): one SVG/PNG anchor with two candidate descriptions.
- `gold` is `a` in the packaged canonical order. Randomize A/B order before presenting examples to a model or human.

`manifest.jsonl` is the authoritative index. Pair details are under `data/<capability>/pairs.jsonl` and `pairs_text.jsonl`. Paths are relative to each capability directory.

Quality-control source:

- `strict_consensus`: all three independent annotators accepted the sample.
- `expert_arbitration`: a small number of samples were added under the same rubric after expert review.

Internal annotator identifiers, raw model reasoning, gate traces, and operational logs are intentionally excluded from this public payload.
