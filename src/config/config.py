"""
Configuration settings for the stance detection project.
"""

# Config file for stance detection - keeps all our settings in one place

import os
from typing import List, Optional


class ModelConfig:
    # FLAN-T5 model settings from documentation
    def __init__(self):
        self.model_name = "google/flan-t5-large"  
        self.max_length = 512  
        self.temperature = 0.1  
        self.do_sample = False  
        self.num_beams = 1  
        self.device = "cpu"  


class DataConfig:
    # Data paths and splitting
    def __init__(self):
        self.raw_data_path = "data/raw_data.csv"  # CSV PATH
        self.processed_data_path = "data/processed/"  # PROCESSED PATH
        self.train_split = 0.7  # 70% for training
        self.val_split = 0.15  # 15% for validation
        self.test_split = 0.15  # 15% for testing
        self.random_state = 42  # For reproducible splits
        self.max_tweet_length = 280  # Twitter limit


class PromptConfig:
    # Different prompts we can try
    def __init__(self):
        self.base_prompt = (
            "What is the stance of the following tweet with respect to COVID-19 vaccine? "
            "Here is the tweet: \"{tweet}\" "
            "Please use exactly one word from the following 3 categories to label it: "
            "\"in-favor\", \"against\", \"neutral-or-unclear\"."
        )
        
        # Some alternative prompts to experiment with
        self.alternative_prompts = [
            "Classify this tweet's stance on COVID-19 vaccination: \"{tweet}\" Options: in-favor, against, neutral-or-unclear",
            "Tweet: \"{tweet}\" What is the author's stance on COVID-19 vaccines? Answer with: in-favor, against, or neutral-or-unclear",
            "Analyze this tweet about COVID-19 vaccination: \"{tweet}\" Choose one: in-favor, against, neutral-or-unclear"
        ]


class TrainingConfig:
    # Fine-tuning settings 
    def __init__(self):
        self.learning_rate = 5e-5  # Pretty standard for transformers
        self.batch_size = 8  # Small batch size for memory
        self.num_epochs = 3  # Don't want to overfit
        self.warmup_steps = 100  
        self.weight_decay = 0.01  
        self.gradient_accumulation_steps = 4  
        self.save_steps = 500  
        self.eval_steps = 500  


class Config:
    # Main config that holds everything
    def __init__(self):
        self.model = ModelConfig()
        self.data = DataConfig()
        self.prompt = PromptConfig()
        self.training = TrainingConfig()
        
        # Where we save 
        self.output_dir = "models/"
        self.results_dir = "results/"
        
        # The three stance categories
        self.stance_labels = ["in-favor", "against", "neutral-or-unclear"]
        
        # Make sure our directories exist
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.data.processed_data_path, exist_ok=True)


# Global config instance - import this everywhere
config = Config()

