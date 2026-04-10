# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Testing suite for the Isaac processor."""

import os
import unittest
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest
from huggingface_hub import hf_hub_download, is_offline_mode

from transformers.models.isaac.processing_isaac import IsaacProcessor
from transformers.testing_utils import require_torch, require_vision
from transformers.utils import is_torch_available, is_vision_available

from ...test_processing_common import ProcessorTesterMixin


if is_torch_available():
    import torch

if is_vision_available():
    from PIL import Image
else:
    Image = None


class IsaacProcessorTestDouble(IsaacProcessor):
    def check_argument_for_proper_class(self, argument_name, argument):
        return type(argument)


def _make_dummy_image(size=(32, 32), color=(255, 0, 0)):
    if Image is None:
        raise RuntimeError("PIL.Image is not available in this environment.")
    return Image.new("RGB", size, color=color)


def _add_generation_tags_to_isaac_chat_template(chat_template: str) -> str:
    if "{% generation %}" in chat_template or "{%- generation" in chat_template:
        return chat_template

    reasoning_block = (
        "{{- '<|im_start|>' + message.role + '\\n<think>\\n' + reasoning_content.strip('\\n') + '\\n</think>\\n\\n' + content.lstrip('\\n') }}"
    )
    patched_reasoning_block = (
        "{{- '<|im_start|>' + message.role + '\\n' }}"
        "{% generation %}"
        "{{- '<think>\\n' + reasoning_content.strip('\\n') + '\\n</think>\\n\\n' + content.lstrip('\\n') }}"
        "{% endgeneration %}"
    )
    plain_assistant_block = "{{- '<|im_start|>' + message.role + '\\n' + content }}"
    patched_plain_assistant_block = (
        "{{- '<|im_start|>' + message.role + '\\n' }}{% generation %}{{- content }}{% endgeneration %}"
    )

    assistant_non_string_block = (
        "        {%- else %}\n"
        "            {{- render_block_turns(message.role, message.content) }}\n"
        "            {%- if message.tool_calls %}"
    )
    patched_assistant_non_string_block = (
        "        {%- else %}\n"
        "            {%- for content in message.content -%}\n"
        "                {%- if content['type'] == 'image' or 'image' in content or 'image_url' in content -%}\n"
        "                    {{- '<|im_start|>' + message.role + '\\n<image><|im_end|>\\n' }}\n"
        "                {%- elif 'text' in content -%}\n"
        "                    {{- '<|im_start|>' + message.role + '\\n' }}{% generation %}{{- content['text'] }}{% endgeneration %}{{- '<|im_end|>\\n' }}\n"
        "                {%- endif -%}\n"
        "            {%- endfor %}\n"
        "            {%- if message.tool_calls %}"
    )

    patched_template = chat_template.replace(reasoning_block, patched_reasoning_block)
    patched_template = patched_template.replace(plain_assistant_block, patched_plain_assistant_block)
    patched_template = patched_template.replace(assistant_non_string_block, patched_assistant_non_string_block)
    return patched_template


BASE_MODEL_ID = os.environ.get("ISAAC_TEST_MODEL_ID", "PerceptronAI/Isaac-0.1-Base")
BASE_MODEL_REVISION = os.environ.get("ISAAC_TEST_MODEL_REVISION", "refs/pr/3") or None
LOCAL_CHECKPOINT = os.environ.get("ISAAC_TEST_MODEL_PATH")


def _checkpoint_or_skip(model_id=BASE_MODEL_ID):
    if LOCAL_CHECKPOINT:
        resolved = Path(LOCAL_CHECKPOINT).expanduser()
        if not resolved.exists():
            pytest.skip(f"Local checkpoint path {resolved} does not exist.")
        return str(resolved)
    if is_offline_mode():
        pytest.skip("Offline mode: set ISAAC_TEST_MODEL_PATH to a local checkpoint to run these tests.")
    return model_id


@lru_cache(maxsize=1)
def _load_chat_template_from_test_revision(model_id=BASE_MODEL_ID, revision=BASE_MODEL_REVISION):
    checkpoint = _checkpoint_or_skip(model_id)
    checkpoint_path = Path(checkpoint)
    if checkpoint_path.exists():
        return checkpoint_path.joinpath("chat_template.jinja").read_text(encoding="utf-8")

    chat_template_path = hf_hub_download(model_id, "chat_template.jinja", revision=revision)
    return Path(chat_template_path).read_text(encoding="utf-8")


@require_torch
@require_vision
class IsaacProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = IsaacProcessorTestDouble
    model_id = BASE_MODEL_ID
    images_input_name = "pixel_values"

    @classmethod
    def _setup_from_pretrained(cls, model_id, **kwargs):
        checkpoint = _checkpoint_or_skip(model_id)
        return super()._setup_from_pretrained(
            checkpoint,
            revision=BASE_MODEL_REVISION,
            patch_size=4,
            max_num_patches=4,
            **kwargs,
        )

    @classmethod
    def _setup_test_attributes(cls, processor):
        cls.image_token = processor.image_token
        cls.pad_token_id = processor.tokenizer.pad_token_id
        cls.image_pad_token_id = processor.image_token_id

    def prepare_image_inputs(self, batch_size: int | None = None, nested: bool = False):
        if batch_size is None:
            return _make_dummy_image(size=(16, 16))
        images = [_make_dummy_image(size=(16, 16), color=(50 * (i + 1), 0, 0)) for i in range(batch_size)]
        if nested:
            return [[image] for image in images]
        return images

    @unittest.skip("Isaac chat templates emit <image> placeholders but the processor consumes image pad tokens")
    def test_apply_chat_template_image_0(self):
        pass

    @unittest.skip("Isaac chat templates emit <image> placeholders but the processor consumes image pad tokens")
    def test_apply_chat_template_image_1(self):
        pass

    def test_apply_chat_template_image_placeholder_expands_to_image_pad_tokens(self):
        processor = self.get_processor()
        image = _make_dummy_image(size=(16, 16))
        messages = [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this."},
                        {"type": "image", "image": image},
                    ],
                }
            ]
        ]

        formatted_prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        self.assertEqual(len(formatted_prompt), 1)
        self.assertIn("<image>", formatted_prompt[0])

        out_dict = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        self.assertTrue(all(key in out_dict for key in ["input_ids", "attention_mask", "pixel_values", "image_grid_thw", "image_metadata", "mm_token_type_ids"]))

        expected_num_image_tokens = processor._get_num_multimodal_tokens(image_sizes=[(image.height, image.width)])["num_image_tokens"][0]
        actual_num_image_tokens = int(out_dict["input_ids"][0].eq(processor.image_token_id).sum().item())

        self.assertEqual(actual_num_image_tokens, expected_num_image_tokens)
        self.assertEqual(int(out_dict["mm_token_type_ids"][0].sum().item()), expected_num_image_tokens)
        self.assertEqual(int(out_dict["image_metadata"][0, 0, 1].item()), expected_num_image_tokens)
        self.assertTrue(torch.all(out_dict["mm_token_type_ids"][0][out_dict["input_ids"][0].eq(processor.image_token_id)] == 1))

    def test_current_chat_template_generation_tags_preserve_text_only_generation_prompt(self):
        processor = self.get_processor()
        patched_template = _add_generation_tags_to_isaac_chat_template(_load_chat_template_from_test_revision())
        messages = [[{"role": "user", "content": [{"type": "text", "text": "What is the capital of France?"}]}]]

        original_prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        patched_prompt = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False, chat_template=patched_template
        )
        self.assertEqual(original_prompt, patched_prompt)

        original_inputs = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
        )
        patched_inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            chat_template=patched_template,
        )
        torch.testing.assert_close(original_inputs["input_ids"], patched_inputs["input_ids"])
        torch.testing.assert_close(original_inputs["attention_mask"], patched_inputs["attention_mask"])

    def test_current_chat_template_generation_tags_preserve_multimodal_generation_prompt(self):
        processor = self.get_processor()
        patched_template = _add_generation_tags_to_isaac_chat_template(_load_chat_template_from_test_revision())
        image = _make_dummy_image(size=(16, 16))
        messages = [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this."},
                        {"type": "image", "image": image},
                    ],
                }
            ]
        ]

        original_prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        patched_prompt = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False, chat_template=patched_template
        )
        self.assertEqual(original_prompt, patched_prompt)

        original_inputs = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
        )
        patched_inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            chat_template=patched_template,
        )
        for key in ["input_ids", "attention_mask", "image_metadata", "pixel_values", "image_grid_thw", "mm_token_type_ids"]:
            torch.testing.assert_close(original_inputs[key], patched_inputs[key])

    def test_current_chat_template_generation_tags_enable_assistant_masks(self):
        processor = self.get_processor()
        patched_template = _add_generation_tags_to_isaac_chat_template(_load_chat_template_from_test_revision())
        messages = [
            [
                {"role": "user", "content": [{"type": "text", "text": "What is the capital of France?"}]},
                {"role": "assistant", "content": [{"type": "text", "text": "The capital of France is Paris."}]},
                {"role": "user", "content": [{"type": "text", "text": "What about Italy?"}]},
                {"role": "assistant", "content": [{"type": "text", "text": "The capital of Italy is Rome."}]},
            ]
        ]

        original_prompt = processor.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)
        patched_prompt = processor.apply_chat_template(
            messages, add_generation_prompt=False, tokenize=False, chat_template=patched_template
        )
        self.assertEqual(original_prompt, patched_prompt)

        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=False,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            return_assistant_tokens_mask=True,
            chat_template=patched_template,
        )
        self.assertGreater(int(inputs["assistant_masks"].sum().item()), 0)

        assistant_ids = inputs["input_ids"][inputs["assistant_masks"].bool()]
        expected_assistant_text = "The capital of France is Paris.The capital of Italy is Rome."
        self.assertEqual(processor.decode(assistant_ids, skip_special_tokens=True), expected_assistant_text)

    def test_get_num_multimodal_tokens_matches_processor_call(self):
        processor = self.get_processor()

        image_sizes = [(100, 100), (300, 100), (500, 30), (213, 167)]
        image_inputs = [np.random.randint(255, size=(h, w, 3), dtype=np.uint8) for h, w in image_sizes]

        text = [f"This is an image {self.image_token}"] * len(image_inputs)
        inputs = processor(
            text=text,
            images=[[image] for image in image_inputs],
            padding=True,
            return_mm_token_type_ids=True,
            return_tensors="pt",
        )

        num_image_tokens_from_call = inputs.mm_token_type_ids.sum(-1).tolist()
        num_image_tokens_from_helper = processor._get_num_multimodal_tokens(image_sizes=image_sizes)
        self.assertListEqual(num_image_tokens_from_call, num_image_tokens_from_helper["num_image_tokens"])
