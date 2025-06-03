from cog import BasePredictor, BaseModel, Path, Input
import torch
from PIL import Image
import open_clip
import io
import numpy as np
from typing import List
import requests

class Output(BaseModel):
    input_type: str
    input: str
    embedding: List[float]

class Predictor(BasePredictor):
    def setup(self):
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            "MobileCLIP-S1",
            pretrained="/weights/open_clip_pytorch_model.bin"
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(
            "MobileCLIP-S1",
            # pretrained="/weights/open_clip_pytorch_model.bin"
        )
        print(f"Model loaded")

    def predict(self,
        image: str = Input(description="Image url to generate embedding for", default=None),
        text: str = Input(description="Text to generate embedding for", default=None)
    ) -> Output:
        """
        Runs a prediction to get the embedding for either an input image or a text string.
        """
        # Ensure exactly one input is provided
        if image is None and text is None:
            raise ValueError("You must provide either an 'image' or a 'text_string' input.")
        if image is not None and text is not None:
            raise ValueError("You must provide either an 'image' or a 'text_string' input, not both.")

        with torch.no_grad():
            if image is not None:
                print(f"Generating embedding for image: {image}")
                # Load the image
                response = requests.get(image, stream=True)
                response.raise_for_status()
                image_data = io.BytesIO(response.content)
                pil_image = Image.open(image_data)

                # Preprocess the image
                image_input = self.preprocess(pil_image).unsqueeze(0)
                image_features = self.model.encode_image(image_input)

                # Normalize features (standard practice for CLIP embeddings)
                image_features /= image_features.norm(dim=-1, keepdim=True)

                # Convert to list for JSON serialization. [0] is used to remove the batch dimension.
                embedding = image_features.cpu().numpy().tolist()[0]

                return {
                    "input_type": "image",
                    "input": image,
                    "embedding": embedding,
                }

            elif text is not None:
                print(f"Generating embedding for text: '{text}'")

                tokens = self.tokenizer([text])
                text_features = self.model.encode_text(tokens)

                # Normalize features
                text_features /= text_features.norm(dim=-1, keepdim=True)

                # Convert to list for JSON serialization. [0] is used to remove the batch dimension.
                embedding = text_features.cpu().numpy().tolist()[0]

                return {
                    "input_type": "image",
                    "input": text,
                    "embedding": embedding,
                }

