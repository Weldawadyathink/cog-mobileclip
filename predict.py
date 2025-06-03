from cog import BasePredictor, Path, Input
import torch
from PIL import Image
import open_clip
import io
import numpy as np

class Predictor(BasePredictor):
    def setup(self):
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            "MobileCLIP-S1",
            pretrained="/weights/open_clip_pytorch_model.bin"
        )
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(
            "MobileCLIP-S1",
            # pretrained="/weights/open_clip_pytorch_model.bin"
        )
        print(f"Model loaded")

    def predict(self,
        image: Path = Input(description="Image url to generate embedding for", default=None),
        text: str = Input(description="Text to generate embedding for", default=None)
    ) -> dict:
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
                with open(image, "rb") as f:
                    pil_image = Image.open(io.BytesIO(f.read())).convert("RGB")

                # Preprocess the image
                image_input = self.preprocess(pil_image).unsqueeze(0).to(self.device)
                image_features = self.model.encode_image(image_input)

                # Normalize features (standard practice for CLIP embeddings)
                image_features /= image_features.norm(dim=-1, keepdim=True)

                # Convert to list for JSON serialization. [0] is used to remove the batch dimension.
                embedding = image_features.cpu().numpy().tolist()[0]

                return {
                    "input_type": "image",
                    "embedding": embedding
                }

            elif text is not None:
                print(f"Generating embedding for text: '{text}'")
                # Tokenize the text prompt
                tokens = self.tokenizer([text])
                # text_features = self.model.encode_text(tokens)

                # Normalize features
                # text_features /= text_features.norm(dim=-1, keepdim=True)

                # Convert to list for JSON serialization. [0] is used to remove the batch dimension.
                # embedding = text_features.cpu().numpy().tolist()[0]

                return {
                    "input_type": "text",
                    "embedding": tokens
                }

