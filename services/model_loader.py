from transformers import pipeline
import torch
from typing import Optional

if torch.cuda.is_available():
    device_id = 0 
    print("CUDA (GPU) is available. Using device 0.")
else:
    device_id = -1
    print("CUDA not available. Using device -1 (CPU).")


class ModelLoader:
    def __init__(self) -> None:
        self.sentiment_model: Optional = None
        self.topic_model: Optional = None
        self.action_model: Optional = None
        self.device: int = device_id

    def load_models(self) -> None:
        print("Loading Sentiment Model...")
        self.sentiment_model = pipeline("text-classification", model="Ysfxjo/marbert-complaint-sentiment", device=self.device, top_k=3)
        print("Loading Topic Model...")
        self.topic_model = pipeline("text-classification", model="Ysfxjo/marbert-saudi-complaint-topic", device=self.device, top_k=3)
        print("Loading Action Model...")
        self.action_model = pipeline("text-classification", model="Ysfxjo/marbert-saudi-complaint-action", device=self.device, top_k=3)
        print("All models loaded successfully!")