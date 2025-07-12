# evaluate the trained model on the hidden test set
# Estimate real-world performance

import pandas as pd
from transformers import T5Tokenizer, T5ForConditionalGeneration
import sys
from tqdm import tqdm
from sklearn.metrics import classification_report, f1_score, accuracy_score
import numpy as np

from ..config.config import config

def load_model():
    print("Loading trained model...")
    model_path = "models/best_model"
    tokenizer = T5Tokenizer.from_pretrained(model_path)
    model = T5ForConditionalGeneration.from_pretrained(model_path)
    print("Model loaded successfully!")
    return tokenizer, model

def predict_stance(tokenizer, model, tweet):
    # Generate stance prediction for a single tweet using the prompt format
    prompt = config.prompt.base_prompt.format(tweet=tweet)
    
    
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    
    
    outputs = model.generate(
        input_ids=inputs['input_ids'],
        attention_mask=inputs['attention_mask']
    )
    prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return prediction.strip()

def main():
    # Main evaluation function - test model on held-out test set
    print("Evaluating model on held-out test set...")
    
    
    tokenizer, model = load_model()
    
    
    print("Loading test data...")
    test_df = pd.read_csv("data/processed/test.csv")
    print(f"Loaded {len(test_df)} test examples")
    
   
    print("Generating predictions...")
    predictions = []
    
    # use tqdm for aesthetics :)
    for tweet in tqdm(test_df['tweet_cleaned'], desc="Processing test tweets"):
        try:
            pred = predict_stance(tokenizer, model, tweet)
            predictions.append(pred)
        except Exception as e:
            print(f"Error processing tweet: {tweet[:50]}... Error: {e}")
            predictions.append("neutral-or-unclear")  # fallback in case of exceptions
    
    # Get the true labels for comparison
    true_labels = test_df['label_majority'].tolist()
    
    # Calculate and display performance metrics
    print("\n" + "="*50)
    print("MODEL PERFORMANCE ON TEST SET")
    print("="*50)
    
    # Overall accuracy
    accuracy = accuracy_score(true_labels, predictions)
    print(f"Overall Accuracy: {accuracy:.3f}")
    
    # F1 score (macro average - treats all classes equally)
    f1_macro = f1_score(true_labels, predictions, average='macro')
    print(f"F1 Score (Macro): {f1_macro:.3f}")
    
    # F1 score (weighted average - accounts for class imbalance)
    f1_weighted = f1_score(true_labels, predictions, average='weighted')
    print(f"F1 Score (Weighted): {f1_weighted:.3f}")
    
    # Detailed breakdown by class
    print("\nDetailed Classification Report:")
    print("-" * 50)
    print(classification_report(true_labels, predictions, target_names=['in-favor', 'against', 'neutral-or-unclear']))
    
    # Show distribution of predictions vs true labels
    print("\nPrediction Distribution:")
    print("-" * 30)
    pred_counts = pd.Series(predictions).value_counts()
    print("Model Predictions:")
    for stance, count in pred_counts.items():
        print(f"  {stance}: {count}")
    
    print("\nTrue Labels:")
    true_counts = pd.Series(true_labels).value_counts()
    for stance, count in true_counts.items():
        print(f"  {stance}: {count}")
    
    # Show some example predictions for manual inspection
    print("\nSample Predictions:")
    print("-" * 30)
    for i in range(min(10, len(test_df))):
        print(f"Tweet: {test_df.iloc[i]['tweet_cleaned'][:100]}...")
        print(f"True: {true_labels[i]}, Predicted: {predictions[i]}")
        print()

if __name__ == "__main__":
    main() 