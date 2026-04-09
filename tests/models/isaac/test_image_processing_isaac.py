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

from transformers.models.isaac.image_processing_isaac import get_image_size_for_max_num_patches
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
        return prepare_image_inputs(
            batch_size=self.batch_size,
            min_resolution=self.min_resolution,
            max_resolution=self.max_resolution,
            num_channels=self.num_channels,
            equal_resolution=equal_resolution,
            numpify=numpify,
            torchify=torchify,
        )

    def expected_output_image_shape(self, images):
        max_patches = 0
        for image in images:
            if isinstance(image, Image.Image):
                width, height = image.size
            elif isinstance(image, np.ndarray):
                height, width = image.shape[:2]
            else:
                height, width = image.shape[-2:]

            target_height, target_width = get_image_size_for_max_num_patches(
                image_height=height,
                image_width=width,
                patch_size=self.patch_size,
                max_num_patches=self.max_num_patches,
                min_num_patches=self.min_num_patches,
                pixel_shuffle_scale=self.pixel_shuffle_scale,
            )
            max_patches = max(max_patches, (target_height // self.patch_size) * (target_width // self.patch_size))

        return (1, max_patches, self.patch_dim)


@require_torch
@require_vision
class IsaacImageProcessingTest(ImageProcessingTestMixin, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.image_processor_tester = IsaacImageProcessingTester(self)

    @property
    def image_processor_dict(self):
        return self.image_processor_tester.prepare_image_processor_dict()

    @unittest.skip(reason="Isaac image processor 4-channel coverage is not defined")
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
            self.assertEqual(tuple(encoding["pixel_values"].shape), (2, 2, 6, 768))

            expected_grids = torch.tensor(
                [
                    [[1, 2, 2], [0, 0, 0]],
                    [[1, 2, 3], [1, 3, 2]],
                ],
                dtype=torch.long,
            )

            torch.testing.assert_close(encoding["image_grid_thw"], expected_grids)
            self.assertTrue(torch.all(encoding["pixel_values"][0, 1] == 0))

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
