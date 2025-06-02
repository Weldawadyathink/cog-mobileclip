import cog
import torch
from PIL import Image
import open_clip
import io
import numpy as np

class Predictor(cog.BasePredictor):
    def setup(self):
        """
        Loads the model and the preprocessor into memory to make running multiple
        predictions efficient.
        """
        print("Loading OpenCLIP model...")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            "MobileCLIP-S0", pretrained="apple"
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval() # Set model to evaluation mode
        print(f"Model loaded on device: {self.device}")

    @cog.input("image", type=cog.Path, help="Input image to analyze")
    @cog.input("text_prompts", type=str,
               help="Comma-separated text prompts (e.g., 'a cat, a dog, a car')")
    def predict(self, image: cog.Path, text_prompts: str) -> dict:
        """
        Runs a single prediction on the model.
        """
        print(f"Processing image: {image}")
        print(f"Text prompts: {text_prompts}")

        # Load the image
        with open(image, "rb") as f:
            pil_image = Image.open(io.BytesIO(f.read())).convert("RGB")

        # Preprocess the image
        image_input = self.preprocess(pil_image).unsqueeze(0).to(self.device)

        # Tokenize the text prompts
        prompts_list = [p.strip() for p in text_prompts.split(',')]
        text_input = open_clip.tokenize(prompts_list).to(self.device)

        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
            text_features = self.model.encode_text(text_input)

            # Normalize features
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            # Calculate cosine similarity
            # Scores will be a tensor of shape (1, num_prompts)
            similarity = (image_features @ text_features.T).squeeze(0)

            # Convert to numpy array and then to a list for JSON serialization
            similarity_scores = similarity.cpu().numpy().tolist()

        # Create a dictionary mapping prompts to their similarity scores
        results = {
            "image_path": str(image),
            "text_prompts": prompts_list,
            "similarity_scores": {prompt: score for prompt, score in zip(prompts_list, similarity_scores)},
            "top_match": prompts_list[np.argmax(similarity_scores)]
        }

        print("Prediction complete.")
        return results

