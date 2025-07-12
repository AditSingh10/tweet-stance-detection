# Test FLAN-T5-Large zero-shot performance locally for comparison


import pandas as pd
import numpy as np
from transformers import T5Tokenizer, T5ForConditionalGeneration
from sklearn.metrics import classification_report, accuracy_score, f1_score
import torch
from src.data.data_loader import DataLoader
from src.config.config import config

def test_zero_shot_performance():
    
    print("=" * 60)
    print("TESTING FLAN-T5-LARGE ZERO-SHOT PERFORMANCE")
    print("=" * 60)
    

    print("\n1. Loading data...")
    loader = DataLoader()
    df = loader.load_data()
    df = loader.clean_data()
    train_df, val_df, test_df = loader.split_data()
    
    print(f"Test samples: {len(test_df)}")
    print(f"Total dataset size: {len(loader.df)}")
    

    print("\n2. Loading FLAN-T5-Large...")
    model_name = "google/flan-t5-large"
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    
    if torch.cuda.is_available():
        model = model.to("cuda")
        print("Using GPU")
    else:
        print("Using CPU")
    

    prompt_template = (
        "Classify the stance of this tweet regarding COVID-19 vaccination. "
        "Tweet: \"{tweet}\" "
        "Choose exactly one label: "
        "• in-favor (explicitly supports vaccination) "
        "• against (explicitly opposes vaccination) "
        "• neutral-or-unclear (neutral, unclear, or not about vaccination) "
        "Label:"
    )
    
    print("\n3. Generating predictions...")
    predictions = []
    true_labels = []
    
    for i, (idx, row) in enumerate(test_df.iterrows()):
        tweet = row['tweet']
        true_label = row['label_majority']
        
        prompt = prompt_template.format(tweet=tweet)
        
        inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=10,
                num_beams=1,
                do_sample=False,
                temperature=0.1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        predicted_text = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        predicted_label = map_prediction_to_label(predicted_text)
        
        predictions.append(predicted_label)
        true_labels.append(true_label)
        
        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{len(test_df)} samples")
    
    print("\n4. Evaluating performance...")
    
    accuracy = accuracy_score(true_labels, predictions)
    f1_macro = f1_score(true_labels, predictions, average='macro')
    f1_weighted = f1_score(true_labels, predictions, average='weighted')
    
    print("\n" + "=" * 60)
    print("ZERO-SHOT PERFORMANCE RESULTS")
    print("=" * 60)
    print(f"Overall Accuracy: {accuracy:.3f}")
    print(f"F1 Score (Macro): {f1_macro:.3f}")
    print(f"F1 Score (Weighted): {f1_weighted:.3f}")
    
    print("\nDetailed Classification Report:")
    print("-" * 50)
    print(classification_report(true_labels, predictions, target_names=config.stance_labels))
    
    print("\nPrediction Distribution:")
    print("-" * 30)
    pred_counts = pd.Series(predictions).value_counts()
    true_counts = pd.Series(true_labels).value_counts()
    
    print("Model Predictions:")
    for label in config.stance_labels:
        count = pred_counts.get(label, 0)
        print(f"  {label}: {count}")
    
    print("\nTrue Labels:")
    for label in config.stance_labels:
        count = true_counts.get(label, 0)
        print(f"  {label}: {count}")
    
    print("\nSample Predictions:")
    print("-" * 30)
    for i in range(min(10, len(test_df))):
        tweet = test_df.iloc[i]['tweet'][:100] + "..." if len(test_df.iloc[i]['tweet']) > 100 else test_df.iloc[i]['tweet']
        print(f"Tweet: {tweet}")
        print(f"True: {true_labels[i]}, Predicted: {predictions[i]}")
        print()

def map_prediction_to_label(predicted_text):
    predicted_text = predicted_text.lower().strip()
    
    if "in-favor" in predicted_text:
        return "in-favor"
    elif "against" in predicted_text:
        return "against"
    elif "neutral-or-unclear" in predicted_text or "neutral" in predicted_text:
        return "neutral-or-unclear"
    
    if "favor" in predicted_text or "support" in predicted_text or "pro" in predicted_text:
        return "in-favor"
    elif "oppose" in predicted_text or "anti" in predicted_text or "against" in predicted_text:
        return "against"
    else:
        return "neutral-or-unclear"  

if __name__ == "__main__":
    test_zero_shot_performance() 