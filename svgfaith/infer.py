"""Run SVGFaith-Score on one prompt and one rendered image.

The released score is three steps:

1. ``Yukinonooo/SVGFaith-Decomposer`` reads only the prompt. Thinking stays
   on, sampling is greedy, and questions are recovered with
   ``greedy_question_salvage_v1``.
2. Unmodified ``Qwen/Qwen3.5-9B`` answers each question on the image with
   thinking off and one generated token. The image is flattened onto white
   and resized to 448. The requirement probability is
   ``P(Yes) / (P(Yes) + P(No))`` under the polarity-aware prompt.
3. ``unique_ancestor_score`` aggregates those probabilities with alpha 0.5.
   Salvage stores 1-based tuple ids; they are converted to list indices here.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from PIL import Image

from svgfaith.aggregate import unique_ancestor_score
from svgfaith.salvage import salvage_questions


PROMPTS = Path(__file__).with_name("prompts")
DECOMPOSER_ID = "Yukinonooo/SVGFaith-Decomposer"
VQA_ID = "Qwen/Qwen3.5-9B"
IMAGE_SIZE = 448
DECOMPOSER_MAX_NEW_TOKENS = 40000

VQA_SYSTEM = (
    "You are a visual requirement verifier. Inspect only the single "
    "provided image. The canonical question has explicit requirement "
    "polarity={polarity}. For present, the fact must be visible; for absent, "
    "the fact must not be visible. Answer Yes when the requirement is "
    "satisfied and No when it is contradicted, missing, or not visibly "
    "supported. Output exactly one word, Yes or No, with no explanation or "
    "punctuation."
)
VQA_USER = (
    "Canonical atomic question: {question}\n"
    "Requirement polarity: {polarity}\n"
    "Answer only Yes or No."
)


def load_text(name: str) -> str:
    return (PROMPTS / name).read_text(encoding="utf-8")


def decomposer_messages(prompt: str) -> list[dict[str, str]]:
    """Build the V2.1 one-demonstration decomposition chat."""
    system = load_text("decomposer_system.txt").strip()
    template = load_text("user_template.txt")
    few_user = json.loads(load_text("few_shot_user.json"))
    few_assistant = json.loads(load_text("few_shot_assistant.json"))
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_message(template, few_user["prompt"])},
        {
            "role": "assistant",
            "content": json.dumps(
                few_assistant,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
        },
        {"role": "user", "content": user_message(template, prompt)},
    ]


def user_message(template: str, prompt: str) -> str:
    if "{{PROMPT_JSON}}" not in template:
        raise ValueError("user template is missing {{PROMPT_JSON}}")
    payload = json.dumps({"prompt": prompt}, ensure_ascii=False, sort_keys=True)
    return template.replace("{{PROMPT_JSON}}", payload)


def vqa_messages(question: str, polarity: str) -> tuple[str, str]:
    """Return the released polarity-aware system and user text."""
    normalized = polarity.strip().lower()
    if normalized not in {"present", "absent"}:
        raise ValueError("polarity must be present or absent")
    system = VQA_SYSTEM.format(polarity=normalized)
    user = VQA_USER.format(question=question, polarity=normalized)
    prompt_system = load_text("vqa_system.txt").strip().format(polarity=normalized)
    prompt_user = load_text("vqa_user.txt").strip().format(
        question=question, polarity=normalized
    )
    if system != prompt_system or user != prompt_user:
        raise RuntimeError("VQA prompt files drifted from the released wording")
    return system, user


def prepare_image(path: Path, image_size: int = IMAGE_SIZE) -> Image.Image:
    """Flatten transparency onto white and resize with Lanczos."""
    image = Image.open(path).convert("RGBA")
    background = Image.new("RGBA", image.size, (255, 255, 255, 255))
    background.alpha_composite(image)
    image = background.convert("RGB")
    if image.size != (image_size, image_size):
        image = image.resize((image_size, image_size), Image.Resampling.LANCZOS)
    return image


def questions_from_generation(content: str) -> list[dict]:
    """Salvage questions and keep the fields VQA and aggregation read."""
    tuples = salvage_questions(content).decomposition["tuples"]
    rows = []
    for atom in tuples:
        question = " ".join(str(atom.get("question") or "").split())
        if not question:
            continue
        polarity = str(atom.get("polarity") or "present").strip().lower()
        if polarity not in {"present", "absent"}:
            polarity = "present"
        parents = []
        for value in atom.get("parents", []):
            if isinstance(value, int) and not isinstance(value, bool):
                parents.append(value)
        rows.append({
            "id": int(atom["id"]),
            "question": question,
            "polarity": polarity,
            "parents": parents,
        })
    if not rows:
        raise ValueError("decomposition produced no questions")
    return rows


def parent_index_lists(questions: list[dict]) -> list[list[int]]:
    """Convert 1-based salvage ids into indices for ``unique_ancestor_score``."""
    by_id = {int(question["id"]): index for index, question in enumerate(questions)}
    lists = []
    for question in questions:
        lists.append([
            by_id[parent]
            for parent in question["parents"]
            if parent in by_id
        ])
    return lists


def binary_probability(logprob_yes: float, logprob_no: float) -> float:
    delta = max(-80.0, min(80.0, logprob_no - logprob_yes))
    return 1.0 / (1.0 + math.exp(delta))


def _single_token_id(tokenizer, text: str) -> int:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) != 1:
        raise ValueError(f"{text!r} is not one tokenizer token: {ids}")
    return ids[0]


def _generate_text(model, tokenizer, messages: list[dict], max_new_tokens: int) -> str:
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {key: value.to(model.device) for key, value in inputs.items()}
    output = model.generate(
        **inputs,
        do_sample=False,
        max_new_tokens=max_new_tokens,
    )
    new_tokens = output[0, inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def _yes_no_probability(model, processor, image: Image.Image, question: str, polarity: str) -> float:
    system, user = vqa_messages(question, polarity)
    messages = [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": user},
            ],
        },
    ]
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        enable_thinking=False,
    )
    inputs = {key: value.to(model.device) for key, value in inputs.items()}
    prompt_length = inputs["input_ids"].shape[-1]
    yes_id = _single_token_id(processor.tokenizer, "Yes")
    no_id = _single_token_id(processor.tokenizer, "No")

    def restrict(input_ids, scores):
        masked = scores.clone()
        masked.fill_(float("-inf"))
        masked[:, yes_id] = scores[:, yes_id]
        masked[:, no_id] = scores[:, no_id]
        return masked

    output = model.generate(
        **inputs,
        do_sample=False,
        max_new_tokens=1,
        logits_processor=[restrict],
        return_dict_in_generate=True,
        output_scores=True,
    )
    if output.sequences.shape[-1] != prompt_length + 1:
        raise RuntimeError("VQA did not generate exactly one token")
    scores = output.scores[0][0]
    return binary_probability(float(scores[yes_id]), float(scores[no_id]))


def score_prompt_and_image(
    prompt: str,
    image_path: Path,
    decomposer,
    decomposer_tokenizer,
    vqa_model,
    vqa_processor,
    max_new_tokens: int = DECOMPOSER_MAX_NEW_TOKENS,
) -> dict:
    """Return the SVGFaith-Score and the per-question probabilities."""
    content = _generate_text(
        decomposer,
        decomposer_tokenizer,
        decomposer_messages(prompt),
        max_new_tokens,
    )
    questions = questions_from_generation(content)
    image = prepare_image(image_path)
    probabilities = [
        _yes_no_probability(
            vqa_model,
            vqa_processor,
            image,
            question["question"],
            question["polarity"],
        )
        for question in questions
    ]
    score = unique_ancestor_score(
        probabilities,
        parent_index_lists(questions),
        alpha=0.5,
    )
    return {
        "score": score,
        "questions": [
            {**question, "probability": probability}
            for question, probability in zip(questions, probabilities)
        ],
    }


def _load_pair(model_id: str, processor: bool):
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.to(device).eval()
    if processor:
        reader = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    else:
        reader = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    return model, reader


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--decomposer", default=DECOMPOSER_ID)
    parser.add_argument("--vqa", default=VQA_ID)
    parser.add_argument("--max-new-tokens", type=int, default=DECOMPOSER_MAX_NEW_TOKENS)
    args = parser.parse_args()
    # The paper run served one 9B model at a time. Unload the decomposer
    # before loading the VQA model so both do not have to stay resident.
    decomposer, tokenizer = _load_pair(args.decomposer, processor=False)
    content = _generate_text(
        decomposer,
        tokenizer,
        decomposer_messages(args.prompt),
        args.max_new_tokens,
    )
    del decomposer, tokenizer
    questions = questions_from_generation(content)
    image = prepare_image(args.image)
    vqa, processor = _load_pair(args.vqa, processor=True)
    probabilities = [
        _yes_no_probability(vqa, processor, image, question["question"], question["polarity"])
        for question in questions
    ]
    result = {
        "score": unique_ancestor_score(
            probabilities,
            parent_index_lists(questions),
            alpha=0.5,
        ),
        "questions": [
            {**question, "probability": probability}
            for question, probability in zip(questions, probabilities)
        ],
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
