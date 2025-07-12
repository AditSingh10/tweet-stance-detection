# Quick test script to verify the trained model works correctly


from transformers import T5Tokenizer, T5ForConditionalGeneration
import sys
import os

from ..config.config import config

def test_model():
    print("Testing trained model...")
    
    # Load the trained model
    model_path = "models/best_model"
    print(f"Loading model from: {model_path}")
    
    tokenizer = T5Tokenizer.from_pretrained(model_path)
    model = T5ForConditionalGeneration.from_pretrained(model_path)
    
    print("Model loaded successfully!")
    
    # Test tweets
    test_tweets = [
        "I got my COVID vaccine today and feel great!",
        "Vaccines are dangerous and cause cancer",
        "What's the weather like today?",
        "I'm not sure about the vaccine, there needs to be more research",
        "Everyone should get vaccinated to protect others. Actually I am joking, DO NOT GET THE VACCINE!"
    ]
    
    print("\nTesting predictions:")
    print("-" * 50)
    
    for i, tweet in enumerate(test_tweets, 1):
        # create prompt using format 
        prompt = config.prompt.base_prompt.format(tweet=tweet)
        
        # tokenize
        inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
        
        # Generate prediction
        outputs = model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask']
        )
        prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        print(f"Test {i}:")
        print(f"Tweet: {tweet}")
        print(f"Prediction: {prediction.strip()}")
        print()

if __name__ == "__main__":
    test_model() 