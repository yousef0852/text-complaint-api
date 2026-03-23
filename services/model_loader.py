from transformers import pipeline
import torch
from typing import Optional
import os
from configs.exceptions import ModelLoadError, ConfigurationError


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

        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            raise ConfigurationError("HF_TOKEN", "Environment variable not found")
        
        try:
            print("Loading Sentiment Model...")
            self.sentiment_model = pipeline(
            "text-classification", 
            model="Ysfxjo/marbert-complaint-sentiment", 
            device=self.device, 
            top_k=3,
            token=hf_token if hf_token else None
            )
            print("Sentiment Model loaded successfully!")
        except OSError as e:
            print(f"Failed to load sentiment model: {e}")
            raise ModelLoadError("sentiment", str(e))
        except Exception as e:
            print(f"Unexpected error loading sentiment model: {e}")
            raise ModelLoadError("sentiment", f"Unexpected error: {str(e)}")
        
        print("Loading Topic Model...")
        try:
            self.topic_model = pipeline(
            "text-classification", 
            model="Ysfxjo/marbert-saudi-complaint-topic", 
            device=self.device, 
            top_k=3,
            token=hf_token if hf_token else None
        )
            print("Topic Model loaded successfully!")
        except Exception as e:
            raise ModelLoadError("topic", str(e))

        print("Loading Action Model...")
        try:
            self.action_model = pipeline(
            "text-classification", 
            model="Ysfxjo/marbert-saudi-complaint-action", 
            device=self.device, 
            top_k=3,
            token=hf_token if hf_token else None
        )
            print("Action Model loaded successfully!")
        except Exception as e:
            raise ModelLoadError("action", str(e))

        print("All models loaded successfully!")