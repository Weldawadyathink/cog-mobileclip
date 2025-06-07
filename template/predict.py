from cog import BasePredictor, BaseModel, Input
import torch
from PIL import Image
import mobileclip
from io import BytesIO
import numpy as np
from typing import List
import requests
import re

class NamedEmbedding(BaseModel):
    input: str
    embedding: List[float]

class Predictor(BasePredictor):
    def setup(self):
        self.model, _, self.preprocess = mobileclip.create_model_and_transforms(
            "${name}",
            pretrained="/weights/${pretrained_file}"
        )
        self.tokenizer = mobileclip.get_tokenizer("${name}")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()

        print(f"Model loaded")

    def embedImageUrl(self, imageUrl):
        print(f"Downloading {imageUrl}")
        image = Image.open(BytesIO(requests.get(imageUrl).content))
        if image.mode in ('RGBA', 'LA', 'P'):
            image = image.convert('RGB')
        with torch.no_grad(), torch.cuda.amp.autocast():
            image = self.preprocess(image).unsqueeze(0)
            embedding = self.model.encode_image(image)
            embedding /= embedding.norm(dim=-1, keepdim=True)
            embedding = embedding.cpu().numpy().tolist()[0]
            return {
                "input": imageUrl,
                "embedding": embedding,
            }

    def embedText(self, text):
        print(f"Embedding text {text}")
        with torch.no_grad(), torch.cuda.amp.autocast():
            tokens = self.tokenizer([text])
            embedding = self.model.encode_text(tokens)
            embedding /= embedding.norm(dim=-1, keepdim=True)
            embedding = embedding.cpu().numpy().tolist()[0]
            return {
                "input": text,
                "embedding": embedding,
            }

    def predict(self,
        inputs: str = Input(description="Newline-separated inputs. Can either be strings of text or image URIs starting with http[s]://", default="No input was provided"),
    ) -> List[NamedEmbedding]:
        """
        Runs a prediction to get the embedding for either an input image or a text string.
        """
        returnval = []

        for line in inputs.strip().splitlines():
            line = line.strip()
            if re.match("^https?://", line):
                returnval.append(self.embedImageUrl(line))
            else:
                result = self.embedText(line)
                returnval.append(result)

        return returnval
