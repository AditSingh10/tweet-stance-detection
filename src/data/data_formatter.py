# Put data in structured format for fine-tuning
# Put the already cleaned tweets into an input/output format so we can train the model

import pandas as pd
from typing import List, Dict, Tuple


from ..config.config import config

class DataFormatter:
    # Put cleaned tweets in structured format for model training

    def __init__(self):
        self.prompt_temp = config.prompt.base_prompt

    def format_for_train(self, df: pd.DataFrame):
        # Return list of dicts with inputs/outputs

        # Check that there is a cleaned version of the tweets text
        if "tweet_cleaned" not in df.columns:
            print("Warning: No 'tweet_cleaned' column found, defaulting to original text")
            tweet_col = 'tweet'
        else:
            tweet_col = 'tweet_cleaned'
        
        if 'label_majority' not in df.columns:
            raise ValueError("No 'label_majority' column found; can't be used for training, check that data was cleaned properly")

        training_examples = []
        
        # Loop through rows
        for idx, row in df.iterrows():
            tweet = row[tweet_col]
            label = row['label_majority']
            
            # Format input using the prompt template
            input_text = self.prompt_temp.format(tweet=tweet)
            
            # Create example
            example = {
                'input': input_text,
                'output': label
            }
            
            training_examples.append(example)
        
        print(f"Formatted {len(training_examples)} training examples")
        return training_examples
    
    def get_training_splits(self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[List, List, List]:
        # Format all three splits for training
        
        train_examples = self.format_for_train(train_df)
        val_examples = self.format_for_train(val_df)
        test_examples = self.format_for_train(test_df)
        
        print(f"Training examples: {len(train_examples)}")
        print(f"Validation examples: {len(val_examples)}")
        print(f"Test examples: {len(test_examples)}")
        
        return train_examples, val_examples, test_examples

def main():
    # Quick test to make sure formatting works
    from .data_loader import DataLoader
    
    # Load and clean data
    loader = DataLoader()
    df = loader.clean_data()
    
    # Split data
    train_df, val_df, test_df = loader.split_data()
    
    # Format for training
    formatter = DataFormatter()
    train_examples, val_examples, test_examples = formatter.get_training_splits(
        train_df, val_df, test_df
    )
    
    # Show a few examples
    print("\nSample training examples:")
    for i, example in enumerate(train_examples[:3]):
        print(f"\nExample {i+1}:")
        print(f"Input: {example['input'][:300]}...")
        print(f"Output: {example['output']}")


if __name__ == "__main__":
    main()
    