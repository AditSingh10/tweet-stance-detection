# Trainer for FLAN-T5 fine-tuning
# Handles training loop, validation, model-saving

import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from datasets import Dataset
import pandas as pd
from typing import List, Dict

from ..config.config import config
from ..data.data_formatter import DataFormatter

class ModelTrainer:
    
    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.formatter = DataFormatter();

    def load_model_tokenizer(self):
        # build so other models can be used (good code practice)
        model_name = config.model.model_name
        print(f"Loading model: {model_name}")

        # get from huggingface
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
    
        print("Model and tokenizer loaded successfully")
        

        pass
    def prepare_dataset(self, examples: List[Dict]):
        # convert to HuggingFace Dataset format
        
       # Generate inputs, targets
        inputs = [example['input'] for example in examples]
        targets = [example['output'] for example in examples]
        
        # Tokenize
        model_inputs = self.tokenizer(
            inputs,
            max_length=config.model.max_length,
            truncation=True,
            padding=True,
            return_tensors="pt"
        )
        
        # Tokenize
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                targets,
                max_length=10, 
                truncation=True,
                padding=True,
                return_tensors="pt"
            )
        
        # Create dataset
        dataset = Dataset.from_dict({
            'input_ids': model_inputs['input_ids'],
            'attention_mask': model_inputs['attention_mask'],
            'labels': labels['input_ids']
        })
        
        return dataset
    def train(self, train_examples: List[Dict], val_examples: List[Dict]):
        # main training funciton
        
        # call prepare datasets
        train_dataset = self.prepare_dataset(train_examples)
        val_dataset = self.prepare_dataset(val_examples)
        
        # Show class distribution for reference
        class_counts = {}
        for example in train_examples:
            label = example['output']
            class_counts[label] = class_counts.get(label, 0) + 1
        
        print(f"Class distribution: {class_counts}")
        print("Training model")
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=config.output_dir,
            learning_rate=config.training.learning_rate,
            per_device_train_batch_size=config.training.batch_size,
            per_device_eval_batch_size=config.training.batch_size,
            num_train_epochs=config.training.num_epochs,
            warmup_steps=config.training.warmup_steps,
            weight_decay=config.training.weight_decay,
            gradient_accumulation_steps=config.training.gradient_accumulation_steps,
            save_steps=config.training.save_steps,
            eval_steps=config.training.eval_steps,
            load_best_model_at_end=config.training.load_best_model_at_end,
            eval_strategy="steps",
            save_strategy="steps",
            metric_for_best_model=config.training.metric_for_best_model,
            greater_is_better=config.training.greater_is_better,
            save_total_limit=config.training.save_total_limit,
            fp16=config.training.fp16,
        )
        
        # Create standard trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
        )
        
        # Train the model
        print("Starting training...")
        trainer.train()
        
        # Save the model
        trainer.save_model(f"{config.output_dir}/best_model")
        self.tokenizer.save_pretrained(f"{config.output_dir}/best_model")
        
        print("Training completed and model saved!")
    