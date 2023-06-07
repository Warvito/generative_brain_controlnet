"""Custom transforms to load non-imaging data."""
from typing import Optional

from monai.config import KeysCollection
from monai.data.image_reader import ImageReader
from monai.transforms.transform import MapTransform, Transform
from transformers import CLIPTokenizer


class ApplyTokenizer(Transform):
    """Transformation to apply the CLIP tokenizer."""

    def __init__(self) -> None:
        self.tokenizer = CLIPTokenizer.from_pretrained("stabilityai/stable-diffusion-2-1-base", subfolder="tokenizer")

    def __call__(self, text_input: str):
        tokenized_sentence = self.tokenizer(
            text_input,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        return tokenized_sentence.input_ids


class ApplyTokenizerd(MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        allow_missing_keys: bool = False,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self._padding = ApplyTokenizer(*args, **kwargs)

    def __call__(self, data, reader: Optional[ImageReader] = None):
        d = dict(data)
        for key in self.key_iterator(d):
            data = self._padding(d[key])
            d[key] = data

        return d
