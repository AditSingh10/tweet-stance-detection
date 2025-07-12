"""
Configuration settings for the stance detection project.
"""

# Config file for stance detection - keeps all our settings in one place

import os
from typing import List, Optional


class ModelConfig:
    # FLAN-T5 model settings from documentation
    def __init__(self):
        self.model_name = "google/flan-t5-large"  # Switch back to large
        self.max_length = 512  # Can use full length with large model
        self.temperature = 0.1  
        self.do_sample = False  
        self.num_beams = 1  
        self.device = "cuda"  # Use GPU


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
            "Classify the stance of this tweet regarding COVID-19 vaccination. "
            "Tweet: \"{tweet}\" "
            "Choose exactly one label, by default pick neutral-or-unclear if you are uncertain: "
            "• in-favor (explicitly supports vaccination) "
            "• against (explicitly opposes vaccination) "
            "• neutral-or-unclear (neutral, unclear, or not about vaccination) "
            "Label:"
        )
        
        # Some alternative prompts to experiment with
        self.alternative_prompts = [
            "Classify this tweet's stance on COVID-19 vaccination: \"{tweet}\" Options: in-favor, against, neutral-or-unclear",
            "Tweet: \"{tweet}\" What is the author's stance on COVID-19 vaccines? Answer with: in-favor, against, or neutral-or-unclear",
            "Analyze this tweet about COVID-19 vaccination: \"{tweet}\" Choose one: in-favor, against, neutral-or-unclear"
        ]


class TrainingConfig:
    # Optimized settings for 5.7K samples - exceptional performance without overfitting
    def __init__(self):
        self.learning_rate = 8e-5  # Slightly higher for better convergence
        self.batch_size = 8  # Larger batch for better gradient estimates
        self.num_epochs = 10  # More training to learn subtle patterns
        self.warmup_steps = 300  # Longer warmup for stability
        self.weight_decay = 0.02  # Balanced regularization
        self.gradient_accumulation_steps = 4  # Effective batch size = 32
        self.save_steps = 150  # More frequent checkpoints
        self.eval_steps = 150  # More frequent evaluation
        self.load_best_model_at_end = True  # Always load best checkpoint
        self.metric_for_best_model = "eval_loss"  # Use loss since F1 not computed automatically
        self.greater_is_better = False  # Lower loss is better
        self.save_total_limit = 2  # Keep only 2 best checkpoints
        self.fp16 = True  # Use mixed precision for efficiency


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

