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


import unittest

import numpy as np

from transformers.testing_utils import require_torch, require_vision
from transformers.utils import is_torch_available, is_vision_available

from ...test_image_processing_common import ImageProcessingTestMixin, prepare_image_inputs


if is_torch_available():
    import torch

if is_vision_available():
    from PIL import Image


def _make_dummy_image(size=(32, 32), color=(255, 0, 0)):
    return Image.new("RGB", size, color=color)


class IsaacImageProcessingTester:
    def __init__(
        self,
        parent,
        batch_size=7,
        num_channels=3,
        min_resolution=30,
        max_resolution=80,
        do_resize=True,
        do_rescale=True,
        rescale_factor=1 / 255,
        do_normalize=True,
        image_mean=None,
        image_std=None,
        patch_size=16,
        max_num_patches=16,
        min_num_patches=4,
        pixel_shuffle_scale=1,
        do_convert_rgb=True,
    ):
        if image_mean is None:
            image_mean = [0.485, 0.456, 0.406]
        if image_std is None:
            image_std = [0.229, 0.224, 0.225]

        self.parent = parent
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.min_resolution = min_resolution
        self.max_resolution = max_resolution
        self.do_resize = do_resize
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.image_mean = image_mean
        self.image_std = image_std
        self.patch_size = patch_size
        self.max_num_patches = max_num_patches
        self.min_num_patches = min_num_patches
        self.pixel_shuffle_scale = pixel_shuffle_scale
        self.do_convert_rgb = do_convert_rgb

    @property
    def patch_dim(self):
        return self.num_channels * self.patch_size * self.patch_size

    def prepare_image_processor_dict(self):
        return {
            "do_resize": self.do_resize,
            "do_rescale": self.do_rescale,
            "rescale_factor": self.rescale_factor,
            "do_normalize": self.do_normalize,
            "image_mean": self.image_mean,
            "image_std": self.image_std,
            "patch_size": self.patch_size,
            "max_num_patches": self.max_num_patches,
            "min_num_patches": self.min_num_patches,
            "pixel_shuffle_scale": self.pixel_shuffle_scale,
            "do_convert_rgb": self.do_convert_rgb,
        }

    def prepare_image_inputs(self, equal_resolution=False, numpify=False, torchify=False):
        images = prepare_image_inputs(
            batch_size=self.batch_size,
            min_resolution=self.min_resolution,
            max_resolution=self.max_resolution,
            num_channels=self.num_channels,
            equal_resolution=equal_resolution,
            numpify=numpify,
            torchify=torchify,
        )
        return [[image] for image in images]


@require_torch
@require_vision
class IsaacImageProcessingTest(ImageProcessingTestMixin, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.image_processor_tester = IsaacImageProcessingTester(self)

    @property
    def image_processor_dict(self):
        return self.image_processor_tester.prepare_image_processor_dict()

    def _assert_output_contract(
        self,
        encoding,
        *,
        expected_batch_size=None,
        expected_max_images=None,
        expected_patch_dim=None,
    ):
        self.assertEqual(set(encoding.keys()), {"pixel_values", "image_grid_thw"})

        pixel_values = encoding["pixel_values"]
        image_grid_thw = encoding["image_grid_thw"]

        if expected_batch_size is None:
            self.assertIsNone(pixel_values)
            self.assertIsNone(image_grid_thw)
            return

        self.assertIsNotNone(pixel_values)
        self.assertIsNotNone(image_grid_thw)
        self.assertEqual(pixel_values.dtype, torch.float32)
        self.assertEqual(image_grid_thw.dtype, torch.long)

        if expected_batch_size is not None:
            self.assertEqual(pixel_values.shape[0], expected_batch_size)
            self.assertEqual(image_grid_thw.shape[0], expected_batch_size)
        if expected_max_images is not None:
            self.assertEqual(pixel_values.shape[1], expected_max_images)
            self.assertEqual(image_grid_thw.shape[1], expected_max_images)
        if expected_patch_dim is not None:
            self.assertEqual(pixel_values.shape[-1], expected_patch_dim)

        self.assertEqual(tuple(image_grid_thw.shape), (pixel_values.shape[0], pixel_values.shape[1], 3))

        active_slots = image_grid_thw[..., 0].eq(1)
        self.assertTrue(torch.all(image_grid_thw[~active_slots].eq(0)))
        self.assertTrue(torch.all(image_grid_thw[active_slots, 1:] > 0))

        expected_patch_counts = image_grid_thw[..., 1] * image_grid_thw[..., 2]
        token_positions = torch.arange(pixel_values.shape[2], device=pixel_values.device).view(1, 1, -1)
        image_patch_attention_mask = active_slots.unsqueeze(-1) & token_positions.lt(
            expected_patch_counts.unsqueeze(-1)
        )

        padded_patch_rows = pixel_values[~image_patch_attention_mask]
        if padded_patch_rows.numel() > 0:
            self.assertTrue(torch.all(padded_patch_rows == 0))

    def test_image_processor_properties(self):
        for image_processing_class in self.image_processing_classes.values():
            image_processor = image_processing_class(**self.image_processor_dict)
            self.assertTrue(hasattr(image_processor, "do_resize"))
            self.assertTrue(hasattr(image_processor, "do_rescale"))
            self.assertTrue(hasattr(image_processor, "rescale_factor"))
            self.assertTrue(hasattr(image_processor, "do_normalize"))
            self.assertTrue(hasattr(image_processor, "image_mean"))
            self.assertTrue(hasattr(image_processor, "image_std"))
            self.assertTrue(hasattr(image_processor, "patch_size"))
            self.assertTrue(hasattr(image_processor, "max_num_patches"))
            self.assertTrue(hasattr(image_processor, "min_num_patches"))
            self.assertTrue(hasattr(image_processor, "pixel_shuffle_scale"))
            self.assertTrue(hasattr(image_processor, "do_convert_rgb"))

    def test_call_pil(self):
        for image_processing_class in self.image_processing_classes.values():
            image_processor = image_processing_class(**self.image_processor_dict)
            image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=False)

            for image in image_inputs:
                self.assertIsInstance(image[0], Image.Image)

            single_output = image_processor(image_inputs[0], return_tensors="pt")
            self._assert_output_contract(
                single_output,
                expected_batch_size=1,
                expected_max_images=1,
                expected_patch_dim=self.image_processor_tester.patch_dim,
            )

            batched_output = image_processor(image_inputs, return_tensors="pt")
            self._assert_output_contract(
                batched_output,
                expected_batch_size=self.image_processor_tester.batch_size,
                expected_max_images=1,
                expected_patch_dim=self.image_processor_tester.patch_dim,
            )

    def test_call_numpy(self):
        for image_processing_class in self.image_processing_classes.values():
            image_processor = image_processing_class(**self.image_processor_dict)
            image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=False, numpify=True)

            for image in image_inputs:
                self.assertIsInstance(image[0], np.ndarray)

            single_output = image_processor(image_inputs[0], return_tensors="pt")
            self._assert_output_contract(
                single_output,
                expected_batch_size=1,
                expected_max_images=1,
                expected_patch_dim=self.image_processor_tester.patch_dim,
            )

            batched_output = image_processor(image_inputs, return_tensors="pt")
            self._assert_output_contract(
                batched_output,
                expected_batch_size=self.image_processor_tester.batch_size,
                expected_max_images=1,
                expected_patch_dim=self.image_processor_tester.patch_dim,
            )

    def test_call_pytorch(self):
        for image_processing_class in self.image_processing_classes.values():
            image_processor = image_processing_class(**self.image_processor_dict)
            image_inputs = self.image_processor_tester.prepare_image_inputs(equal_resolution=False, torchify=True)

            for image in image_inputs:
                self.assertIsInstance(image[0], torch.Tensor)

            single_output = image_processor(image_inputs[0], return_tensors="pt")
            self._assert_output_contract(
                single_output,
                expected_batch_size=1,
                expected_max_images=1,
                expected_patch_dim=self.image_processor_tester.patch_dim,
            )

            batched_output = image_processor(image_inputs, return_tensors="pt")
            self._assert_output_contract(
                batched_output,
                expected_batch_size=self.image_processor_tester.batch_size,
                expected_max_images=1,
                expected_patch_dim=self.image_processor_tester.patch_dim,
            )

    @unittest.skip(reason="Isaac image processor 4-channel coverage is not defined yet")
    def test_call_numpy_4_channels(self):
        pass

    def test_nested_multi_image_batch_preserves_grids_and_padding(self):
        for image_processing_class in self.image_processing_classes.values():
            image_processor = image_processing_class(
                **{
                    **self.image_processor_dict,
                    "do_resize": False,
                    "patch_size": 16,
                    "max_num_patches": 64,
                    "min_num_patches": 1,
                    "pixel_shuffle_scale": 1,
                }
            )
            image_inputs = [
                [_make_dummy_image(size=(32, 32), color=(255, 0, 0))],
                [
                    _make_dummy_image(size=(48, 32), color=(0, 255, 0)),
                    _make_dummy_image(size=(32, 48), color=(0, 0, 255)),
                ],
            ]

            encoding = image_processor(image_inputs, return_tensors="pt")
            self._assert_output_contract(
                encoding,
                expected_batch_size=2,
                expected_max_images=2,
                expected_patch_dim=self.image_processor_tester.patch_dim,
            )
            self.assertEqual(tuple(encoding["pixel_values"].shape), (2, 2, 6, 768))

            expected_grids = torch.tensor(
                [
                    [[1, 2, 2], [0, 0, 0]],
                    [[1, 2, 3], [1, 3, 2]],
                ],
                dtype=torch.long,
            )

            torch.testing.assert_close(encoding["image_grid_thw"], expected_grids)

    def test_all_empty_images_returns_none_visual_fields(self):
        for image_processing_class in self.image_processing_classes.values():
            image_processor = image_processing_class(**self.image_processor_dict)
            encoding = image_processor([[], []], return_tensors="pt")

            self._assert_output_contract(encoding, expected_batch_size=None)

    def test_do_resize_false_requires_patch_divisibility(self):
        for image_processing_class in self.image_processing_classes.values():
            image_processor = image_processing_class(
                **{
                    **self.image_processor_dict,
                    "do_resize": False,
                    "patch_size": 16,
                }
            )

            with self.assertRaisesRegex(ValueError, "must be divisible by patch_size"):
                image_processor([[_make_dummy_image(size=(31, 32))]], return_tensors="pt")

    def test_pixel_shuffle_scale_requires_divisible_token_grid(self):
        for image_processing_class in self.image_processing_classes.values():
            image_processor = image_processing_class(
                **{
                    **self.image_processor_dict,
                    "do_resize": False,
                    "patch_size": 16,
                    "pixel_shuffle_scale": 2,
                }
            )

            with self.assertRaisesRegex(ValueError, "must be divisible by pixel_shuffle_scale"):
                image_processor([[_make_dummy_image(size=(32, 16))]], return_tensors="pt")

