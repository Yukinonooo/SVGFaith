# SVGFaith-Arena

1,327 human comparisons of real text-to-SVG outputs. In every row, `image_a` is the candidate judged more faithful to `prompt`, so `gold` is `a`.

`label_source` is `majority_vote` (1,288 pairs) or `expert_arbitration` (39 pairs) after the published label corrections: 12 reversed pairs and 25 removed pairs relative to the 1,352-pair pre-correction pool. Renderings of removed pairs are not included.

The six generators are OmniSVG1.1-8B, InternSVG-8B, SVGen (Qwen2.5-Coder-7B), GPT-5.4, Gemini-3.1-Pro, and Claude-Opus-4.6. `winner_model` produced `image_a`.
