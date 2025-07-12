# COVID-19 Vaccination Stance Detection

This repository contains an ML solution for detecting stance on COVID-19 vaccination from Twitter data. The project fine-tunes Google's FLAN-T5-Large model to classify tweets into three categories: "in-favor", "against", and "neutral-or-unclear".

The fine-tuned weights are available upon request. Due to their size, they are not included in the repository.

## Project Overview

### Task
Predict the stance of 5,751 tweets regarding COVID-19 vaccination using a finetuned FLAN-T5-Large language model. The stance classification task involves determining whether a tweet supports, opposes, or is neutral/unclear about vaccination.

### Dataset
- **Size**: 5,751 tweets
- **Labels**: 3 classes (in-favor, against, neutral-or-unclear)
- **Format**: CSV with tweet text and human-annotated ground truth labels
- **Split**: 70% train, 15% validation, 15% test (stratified)

## Technical Approach

### Model Architecture
- **Base Model**: Google FLAN-T5-Large (770M parameters)
- **Framework**: HuggingFace Transformers + PyTorch
- **Training**: Fine-tuning with custom prompt engineering

### Prompt Engineering
The model uses structured prompts to guide stance classification:
```
Analyze this tweet's stance on COVID-19 vaccination.
Tweet: "{tweet}"
If the tweet clearly supports vaccination, answer: in-favor.
If the tweet clearly opposes vaccination, answer: against.
If the tweet is neutral, unclear, or not about vaccination, answer: neutral-or-unclear.
Answer with only the label: in-favor, against, or neutral-or-unclear.
```

### Training Strategy

- **Learning Rate**: 8e-5
- **Batch Size**: 8 (effective batch size: 32 with gradient accumulation)
- **Epochs**: 10
- **Optimizer**: AdamW with weight decay
- **Mixed Precision**: FP16 for efficiency
- **Early Stopping**: Based on validation loss

## Results

### Performance Comparison

| Approach | F1 Macro | F1 Weighted | Accuracy |
|----------|----------|-------------|----------|
| Zero-shot FLAN-T5-Large | **0.427** | **0.534** | **0.594** |
| Fine-tuned FLAN-T5-Large | **0.612** | **0.697** | **0.731** |

### Detailed Results (Zero-shot)
```
                    precision    recall  f1-score   support
          in-favor       0.56      0.58      0.57       271
           against       0.61      0.81      0.70       436
neutral-or-unclear       1.00      0.01      0.01       156
```

**Prediction Distribution:**
- Model Predictions: in-favor (578), against (284), neutral-or-unclear (1)
- True Labels: in-favor (271), against (436), neutral-or-unclear (156)

### Detailed Results (Fine-tuned)
```
                    precision    recall  f1-score   support
          in-favor       0.75      0.76      0.76       271
           against       0.73      0.91      0.81       436
neutral-or-unclear       0.57      0.17      0.27       156
```

### Key Findings
- **Significant improvement** from zero-shot to fine-tuned (F1 Macro: 0.427 → 0.612, +43% improvement)
- **Zero-shot bias** - model heavily overpredicts "in-favor" (578 vs 271 true) and underpredicts "neutral-or-unclear" (1 vs 156 true)
- **Strong fine-tuned performance** on majority classes (in-favor, against)
- **Class imbalance challenge** - neutral-or-unclear class severely underpredicted in both zero-shot and fine-tuned
- **Model bias** toward majority classes due to imbalanced training data

## Project Structure

```
stance/
├── data/
│   ├── raw_data.csv              # Original dataset
│   └── processed/                # Train/val/test splits
├── src/
│   ├── config/
│   │   └── config.py            # Configuration settings
│   ├── data/
│   │   ├── data_loader.py       # Data loading and preprocessing
│   │   └── data_formatter.py    # Prompt formatting
│   └── models/
│       ├── trainer.py           # Model training pipeline
│       └── evaluate_model.py    # Model evaluation
├── models/                      
├── results/                     
```

## Installation & Usage

### Requirements
```bash
pip install -r requirements.txt
```

### Training
```bash
python -m src.models.trainer
```

### Evaluation
```bash
python -m src.models.evaluate_model
```

### Zero-shot Testing
```bash
python -m src.models.test_zero_shot
```

## Challenges & Next Steps

### Current Limitations
1. **Class Imbalance**: Neutral-or-unclear class severely underpredicted
2. **Model Bias**: Tendency to predict majority classes


### Proposed Improvements
1. **Data Augmentation**: Create synthetic neutral-or-unclear examples
2. **Oversampling**: Balance training data through strategic sampling
3. **Class Weights**

## Technical Details

### Training Configuration
- **Hardware**: NVIDIA V100 GPU (Colab Pro)
- **Training Time**: ~1-2 hours for 10 epochs
- **Memory Usage**: ~16GB GPU memory
- **Checkpointing**: Best model saved based on validation loss

### Evaluation Metrics
- **Primary**: F1 Macro (accounts for class imbalance)
- **Secondary**: F1 Weighted, Accuracy, Per-class F1 scores
- **Analysis**: Confusion matrix, prediction distribution

## Conclusion

This project successfully demonstrates the effectiveness of fine-tuning large language models for stance detection tasks. The 23% improvement in accuracy (0.594 → 0.731) shows the value of task-specific training. The zero-shot baseline reveals significant model bias toward the "in-favor" class, which fine-tuning helps address. However, the class imbalance challenge persists and highlights the need for more sophisticated data balancing techniques in future developments of this project.

