
#Generate predictions for all tweets in the dataset using the fine-tuned FLAN-T5 model.

import pandas as pd
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from tqdm import tqdm
import os
import sys

from ..config.config import config

def load_model_and_tokenizer():
    """Load the fine-tuned model and tokenizer."""
    print("Loading fine-tuned model and tokenizer...")
    
    # make sure fine-tuned model exists
    model_path = "models/best_model"
    if not os.path.exists(model_path):
        print(f"Fine-tuned model not found at {model_path}")
        return None, None
    
    try:
        tokenizer = T5Tokenizer.from_pretrained(model_path)
        model = T5ForConditionalGeneration.from_pretrained(model_path)
        
        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        
        print(f"Model loaded successfully on {device}")
        return model, tokenizer, device
        
    except Exception as e:
        print(f"Error loading fine-tuned model: {e}")
        return None, None, None

def create_prompt(tweet):
    """Create the prompt for stance classification using config."""
    return config.prompt.base_prompt.format(tweet=tweet)

def predict_stance(model, tokenizer, tweet, device):
    """Predict stance for a single tweet."""
    prompt = create_prompt(tweet)
    
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Generate prediction
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=10,
            num_beams=1,
            do_sample=False,
            early_stopping=True
        )
    
    # Decode prediction
    prediction = tokenizer.decode(outputs[0], skip_special_tokens=True).strip().lower()
    
    # Clean up prediction
    if "in-favor" in prediction:
        return "in-favor"
    elif "against" in prediction:
        return "against"
    elif "neutral" in prediction or "unclear" in prediction:
        return "neutral-or-unclear"
    else:
        # Fallback based on common patterns
        if any(word in tweet.lower() for word in ["get vaccinated", "vaccine works", "get the shot", "protect yourself"]):
            return "in-favor"
        elif any(word in tweet.lower() for word in ["anti-vax", "vaccine passport", "mandate", "forced"]):
            return "against"
        else:
            return "neutral-or-unclear"

def main():
    """Main function to generate predictions for all tweets."""
    print("=" * 60)
    print("COVID-19 VACCINATION STANCE PREDICTION")
    print("=" * 60)
    
    # Load model and tokenizer
    model, tokenizer, device = load_model_and_tokenizer()
    if model is None:
        print("Failed to load model. Exiting.")
        return
    
    # Load dataset
    csv_file = "Q2_20230202_majority 1.csv"
    print(f"Loading dataset from {csv_file}...")
    
    try:
        df = pd.read_csv(csv_file)
        print(f"Loaded {len(df)} tweets")
        print(f"Columns: {list(df.columns)}")
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return
    
    # Check if label_pred column already exists
    if 'label_pred' in df.columns:
        print("Warning: label_pred column already exists. Overwriting...")
        df = df.drop('label_pred', axis=1)
    
    # Generate predictions
    print("\nGenerating predictions...")
    predictions = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Predicting stances"):
        tweet = row['tweet']
        prediction = predict_stance(model, tokenizer, tweet, device)
        predictions.append(prediction)
    
    # Add predictions to dataframe
    df['label_pred'] = predictions
    
    # Show prediction distribution
    print("\nPrediction Distribution:")
    pred_counts = df['label_pred'].value_counts()
    for stance, count in pred_counts.items():
        print(f"  {stance}: {count}")
    
    # Save updated CSV
    output_file = "Q2_20230202_majority_with_predictions.csv"
    print(f"\nSaving predictions to {output_file}...")
    
    try:
        df.to_csv(output_file, index=False)
        print(f"Successfully saved {len(df)} predictions to {output_file}")
    except Exception as e:
        print(f"Error saving file: {e}")
        return
    
    # Show sample predictions
    print("\nSample Predictions:")
    print("-" * 80)
    sample_df = df[['tweet', 'label_majority', 'label_pred']].head(10)
    for idx, row in sample_df.iterrows():
        tweet = row['tweet'][:100] + "..." if len(row['tweet']) > 100 else row['tweet']
        print(f"Tweet: {tweet}")
        print(f"True: {row['label_majority']} | Predicted: {row['label_pred']}")
        print("-" * 40)
    
    print("\nPrediction generation complete!")
    print(f"Output file: {output_file}")

if __name__ == "__main__":
    main() 