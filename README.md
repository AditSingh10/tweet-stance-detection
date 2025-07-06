# COVID-19 Vaccination Stance Detection

This project classifies tweets about COVID-19 vaccination into three categories: "in-favor", "against", and "neutral-or-unclear". Uses Google's FLAN-T5-Large model with fine-tuning for improved performance.

## Project Structure

```
stance/
├── data/               # Data files
│   ├── raw_data.csv   # Original dataset
│   └── processed/     # Processed data
├── src/               # Source code
│   ├── data/          # Data processing modules
│   ├── models/        # Model implementations
│   ├── utils/         # Utility functions
│   └── config/        # Configuration files
├── models/            # Saved model checkpoints
├── notebooks/         # Jupyter notebooks for exploration
├── tests/             # Unit tests
└── requirements.txt   # Python dependencies
```

## Getting Started

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Download the dataset:**
   - Drop the CSV file in `data/raw_data.csv`

3. **Run the stance detection:**
   ```bash
   python src/main.py
   ```

## Model Architecture

- **Base Model**: Google FLAN-T5-Large
- **Approach**: Zero-shot classification with prompt engineering
- **Optional**: Fine-tuning for better performance

## Dataset
- **Size**: 5,751 tweets
- **Classes**: 3 (in-favor, against, neutral-or-unclear)
- **Evaluation**: F1 Score on held-out test set

## Quick Example

## License

This project is for research purposes only. 