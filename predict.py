from cog import BasePredictor, Path, Input
import torch
from PIL import Image
import open_clip
import io
import numpy as np

class Predictor(BasePredictor):
    def setup(self):
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            "MobileCLIP-S0", pretrained="apple"
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()
        print(f"Model loaded on device: {self.device}")

    def predict(self,
        image: Path = Input(description="Image url to generate embedding for", default=None),
        text_string: str = Input(description="Text to generate embedding for", default=None)
    ) -> dict:
        """
        Runs a prediction to get the embedding for either an input image or a text string.
        """
        # Ensure exactly one input is provided
        if image is None and text_string is None:
            raise ValueError("You must provide either an 'image' or a 'text_string' input.")
        if image is not None and text_string is not None:
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

            elif text_string is not None:
                print(f"Generating embedding for text: '{text_string}'")
                # Tokenize the text prompt
                text_input = open_clip.tokenize([text_string]).to(self.device)
                text_features = self.model.encode_text(text_input)

                # Normalize features
                text_features /= text_features.norm(dim=-1, keepdim=True)

                # Convert to list for JSON serialization. [0] is used to remove the batch dimension.
                embedding = text_features.cpu().numpy().tolist()[0]

                return {
                    "input_type": "text",
                    "embedding": embedding
                }

