import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from svgfaith.infer import (
    decomposer_messages,
    parent_index_lists,
    prepare_image,
    questions_from_generation,
    vqa_messages,
)
from svgfaith.salvage import salvage_questions


class InferenceProtocolTest(unittest.TestCase):
    def test_vqa_prompt_matches_released_wording(self) -> None:
        system, user = vqa_messages("Is the apple red?", "absent")
        self.assertIn("polarity=absent", system)
        self.assertTrue(user.startswith("Canonical atomic question: Is the apple red?"))
        self.assertIn("Requirement polarity: absent", user)

    def test_decomposer_chat_has_one_demonstration(self) -> None:
        messages = decomposer_messages("A red apple.")
        self.assertEqual([item["role"] for item in messages], ["system", "user", "assistant", "user"])
        self.assertIn('{"prompt": "A red apple."}', messages[-1]["content"])
        demonstration = json.loads(messages[2]["content"])
        self.assertIn("tuples", demonstration)

    def test_salvage_remaps_parent_ids_to_indices(self) -> None:
        content = json.dumps({
            "tuples": [
                {"id": 4, "question": "Is there an apple?", "parents": [], "polarity": "present"},
                {"id": 9, "question": "Is the apple red?", "parents": [4], "polarity": "present"},
            ]
        })
        questions = questions_from_generation(content)
        self.assertEqual([question["id"] for question in questions], [1, 2])
        self.assertEqual(parent_index_lists(questions), [[], [0]])

    def test_salvage_recovers_a_truncated_question(self) -> None:
        content = '{"tuples":[{"question":"Is there a cat?"}, {"question":"Is the'
        result = salvage_questions(content)
        self.assertEqual(result.decomposition["tuples"][0]["question"], "Is there a cat?")

    def test_prepare_image_is_white_square(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tiny.png"
            Image.new("RGBA", (2, 2), (0, 0, 0, 0)).save(path)
            image = prepare_image(path, image_size=8)
        self.assertEqual(image.size, (8, 8))
        self.assertEqual(image.getpixel((0, 0)), (255, 255, 255))


if __name__ == "__main__":
    unittest.main()
