# Data loader for the COVID stance detection project
# This handles loading the CSV, cleaning it up, and splitting into train/val/test

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, Optional
import re
from pathlib import Path

from ..config.config import config


class DataLoader:
    # Main class for loading and processing our tweet data
    # Pretty straightforward - load CSV, clean it, split it
    
    def __init__(self, data_path: Optional[str] = None):
        # Initialize with path to our data file
        # If no path given, use the one from config
        self.data_path = data_path or config.data.raw_data_path
        self.df = None  # Will hold our main dataframe
        self.train_df = None  # Training split
        self.val_df = None  # Validation split  
        self.test_df = None  # Test split
        
    def load_data(self) -> pd.DataFrame:
        # Load the CSV file into a pandas dataframe
        # Returns the loaded dataframe
        print(f"Loading data from {self.data_path}")
        
        try:
            self.df = pd.read_csv(self.data_path)
            print(f"Successfully loaded {len(self.df)} rows")
            return self.df
        except FileNotFoundError:
            print(f"File not found: {self.data_path}")
            raise
        except Exception as e:
            print(f"Error loading data: {e}")
            raise
    
    def clean_data(self) -> pd.DataFrame:
        # Clean up the data - handle missing values, clean text, etc.
        # This is where we do most of the preprocessing work
        if self.df is None:
            self.load_data()
        
        print("Starting data cleaning process")
        
        # Check what's missing first
        initial_missing = self.df.isnull().sum()
        print(f"Missing values before cleaning:\n{initial_missing}")
        
        # Drop rows where we don't have the actual tweet text
        self.df = self.df.dropna(subset=['tweet'])
        
        # If we have labels but some are missing, fill with neutral
        if 'label_majority' in self.df.columns:
            self.df['label_majority'] = self.df['label_majority'].fillna('neutral-or-unclear')
        
        # Clean up the tweet text
        self.df['tweet_cleaned'] = self.df['tweet'].apply(self._clean_text)
        
        # Get rid of tweets that are too short after cleaning
        self.df = self.df[self.df['tweet_cleaned'].str.len() > 10]
        
        # See how much we cleaned up
        final_missing = self.df.isnull().sum()
        print(f"Missing values after cleaning:\n{final_missing}")
        print(f"Final dataset size: {len(self.df)} rows")
        
        return self.df
    
    def _clean_text(self, text: str) -> str:
        # Clean up individual tweet text
        # Remove URLs, mentions, hashtags, extra spaces, etc.
        if pd.isna(text):
            return ""
        
        # Make sure it's a string
        text = str(text)
        
        # Get rid of URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove @ mentions
        text = re.sub(r'@\w+', '', text)
        
        # Keep hashtag text but remove the # symbol
        text = re.sub(r'#(\w+)', r'\1', text)
        
        # Clean up multiple spaces
        text = re.sub(r'\s+', ' ', text)
        
        # Trim whitespace
        text = text.strip()
        
        return text
    
    def analyze_class_distribution(self) -> Dict[str, int]:
        # Look at how many examples we have for each stance category
        # Returns a dict with counts for each label
        
        label_counts = self.df['label_majority'].value_counts().to_dict()
        
        # Print out the distribution
        print("Class distribution:")
        for label, count in label_counts.items():
            percentage = (count / len(self.df)) * 100
            print(f"  {label}: {count} ({percentage:.1f}%)")
        
        return label_counts
    
    def split_data(self, 
                   train_size: float = 0.7, 
                   val_size: float = 0.15, 
                   test_size: float = 0.15,
                   random_state: int = 42):
        # Split our data into train/validation/test sets
        # Uses stratified splitting to keep class balance
        # Make sure our splits add up to 1.0
        total = train_size + val_size + test_size
        if abs(total - 1.0) > 0.0001: # account for float error
            raise ValueError(f"Split proportions must sum to 1.0, got {total}")
        
        print(f"Splitting data: train={train_size}, val={val_size}, test={test_size}")
        
        # First split off the test set
        train_val_df, self.test_df = train_test_split(
            self.df, 
            test_size=test_size, 
            random_state=random_state,
            stratify=self.df['label_majority'] if 'label_majority' in self.df.columns else None
        )
        
        # Then split the rest into train and validation
        val_ratio = val_size / (train_size + val_size)
        self.train_df, self.val_df = train_test_split(
            train_val_df,
            test_size=val_ratio,
            random_state=random_state,
            stratify=train_val_df['label_majority'] if 'label_majority' in train_val_df.columns else None
        )
        
        # Log the final split sizes
        print(f"Split complete:")
        print(f"  Train: {len(self.train_df)} samples")
        print(f"  Validation: {len(self.val_df)} samples")
        print(f"  Test: {len(self.test_df)} samples")
        
        return self.train_df, self.val_df, self.test_df
    
    def save_splits(self, output_dir: Optional[str] = None) -> None:
        # Save our train/val/test splits to CSV files
        if self.train_df is None or self.val_df is None or self.test_df is None:
            raise ValueError("Data splits not created yet. Call split_data() first.")
        
        output_dir = output_dir or config.data.processed_data_path
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Write the splits to files
        self.train_df.to_csv(f"{output_dir}/train.csv", index=False)
        self.val_df.to_csv(f"{output_dir}/val.csv", index=False)
        self.test_df.to_csv(f"{output_dir}/test.csv", index=False)
        
        print(f"Data splits saved to {output_dir}")


def main():
    # Quick test to make sure everything works
    # Run this to process the data and see what we get
    loader = DataLoader()
    
    # Load and clean everything
    df = loader.clean_data()
    
    # See how the labels are distributed
    distribution = loader.analyze_class_distribution()
    
    # Split into train/val/test
    train_df, val_df, test_df = loader.split_data()
    
    # Save the splits for later
    loader.save_splits()


if __name__ == "__main__":
    main() 