# Product Review Sentiment Analysis

A complete sentiment analysis pipeline using transformer models (DistilBERT) to classify product reviews into positive, neutral, or negative categories.

## 🎯 Project Overview

This project implements a sentiment analysis system for product reviews using both traditional machine learning (TF-IDF + Logistic Regression) and modern transformer-based approaches (DistilBERT). The system includes data preprocessing, model training, evaluation, and a Streamlit demo interface.

## ✨ Features

- **Multi-class sentiment classification** (positive, neutral, negative)
- **Balanced dataset** with proper data cleaning and preprocessing
- **Baseline model** using TF-IDF and Logistic Regression
- **Transformer model** using DistilBERT for state-of-the-art performance
- **Interactive demo** with Streamlit web interface
- **Comprehensive evaluation** with metrics and visualizations

## 📋 Requirements

```
streamlit
torch
transformers
pandas
numpy
scikit-learn
matplotlib
seaborn
datasets
evaluate
```

## 🚀 Installation

1. Clone the repository:
```bash
git clone https://github.com/marieem3/reviews-analysis-project.git
cd reviews-analysis-project
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the dataset:
   - Download the Amazon Fine Food Reviews dataset from [Kaggle](https://www.kaggle.com/snap/amazon-fine-food-reviews)
   - Place `Reviews.csv` in the project root directory
   - **Note**: The CSV file is not included in the repository due to its large size (>280MB)
   - Dataset should contain `Text` and `Score` columns

## 💻 Usage

### Training the Model

Run the complete training pipeline:

```bash
python sentiment_analysis.py
```

This script will:
1. Load and clean the dataset
2. Perform exploratory data analysis
3. Train a baseline TF-IDF model
4. Train the DistilBERT transformer model
5. Save the trained model to `distilbert_sentiment_model/`

### Running the Demo

Launch the Streamlit web interface:

```bash
streamlit run app.py
```

The demo will be available at `http://localhost:8501`

## 🎥 Demo

Watch the live demo of the Streamlit application in action:

[![Demo Video](https://img.youtube.com/vi/-cFvx5xMM0I/maxresdefault.jpg)](https://www.youtube.com/watch?v=-cFvx5xMM0I)

**Click the image above to watch the demo video**

**Demo Features**:
- Enter any product review text
- Get instant sentiment prediction (positive/neutral/negative)
- View confidence scores
- Clean and intuitive interface

## 📊 Pipeline Details

### 1. Data Preprocessing
- Handles missing values and invalid entries
- Maps 1-5 star ratings to sentiment labels:
  - 1-2 stars → negative
  - 3 stars → neutral
  - 4-5 stars → positive
- Balances dataset with 3,333 samples per class
- Cleans text (removes URLs, HTML tags, special characters)
- Removes duplicates and short reviews

### 2. Exploratory Data Analysis
- Label distribution visualization
- Text length analysis
- Top words per sentiment category
- Sample review examples

### 3. Baseline Model
- **Method**: TF-IDF vectorization + Logistic Regression
- **Features**: 5,000 TF-IDF features with bigrams
- **Purpose**: Establish performance baseline

### 4. Transformer Model
- **Architecture**: DistilBERT (distilbert-base-uncased)
- **Max sequence length**: 128 tokens
- **Training**: 2 epochs with batch size 8
- **Optimizer**: AdamW with learning rate 2e-5
- **Metric**: Weighted F1-score

### 5. Model Evaluation
- Classification report with precision, recall, F1-score
- Confusion matrix visualization
- Per-class performance metrics

## 📁 Project Structure

```
sentiment-analysis-project/
├── sentiment_analysis.py      # Main training pipeline
├── app.py                      # Streamlit demo application
├── Reviews.csv                 # Input dataset (download separately)
├── clean_reviews.csv           # Cleaned and preprocessed data
├── distilbert_sentiment_model/ # Saved model directory
│   ├── config.json
│   ├── model.safetensors
│   ├── tokenizer files
│   └── label_mappings.json
├── results/                    # Training checkpoints
├── requirements.txt            # Python dependencies
├── .gitignore                  # Excludes large files
└── README.md                   # This file
```

## ⚠️ Important Notes

- **Large files excluded**: `Reviews.csv`, `clean_reviews.csv`, and the `distilbert_sentiment_model/` directory are not tracked in git due to their size
- Users must download the dataset separately from Kaggle
- Pre-trained models can be shared via Google Drive or other file hosting services

## 🎯 Model Performance

The DistilBERT model achieves high performance on balanced sentiment classification:
- **Positive reviews**: High precision and recall
- **Negative reviews**: Strong classification accuracy
- **Neutral reviews**: Good separation from positive/negative

Detailed metrics are displayed during training and evaluation.

## 🖥️ Demo Interface

The Streamlit app provides:
- Text area for entering product reviews
- Real-time sentiment prediction
- Confidence score display
- Clean and intuitive interface

## 🔧 Configuration

Key parameters in `sentiment_analysis.py`:
- `max_features=5000`: TF-IDF feature limit
- `max_length=128`: Maximum token length for DistilBERT
- `num_train_epochs=2`: Training epochs
- `learning_rate=2e-5`: Learning rate for fine-tuning
- `per_device_train_batch_size=8`: Batch size

## 📝 Notes

- The model is trained on balanced data (equal samples per class)
- Text cleaning removes URLs, HTML tags, and special characters
- Training progress is logged every 50 steps
- Best model is automatically saved based on F1-score
- GPU acceleration is supported if available

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## 📄 License

This project is open source and available under the MIT License.

## 👨‍💻 Author

**Mariam Jmal**

📧 Email: jmalmariam3@gmail.com

💼 LinkedIn: [Maryem Jmal](https://www.linkedin.com/in/maryem-jmal-b5094b37b)

## 🙏 Acknowledgments

- Hugging Face Transformers library
- DistilBERT model by Hugging Face
- Streamlit for the demo interface
- Amazon Product Reviews dataset

---

**Note**: The `Reviews.csv` dataset is not included in the repository due to its large size (>280MB). Please download it from [Kaggle](https://www.kaggle.com/snap/amazon-fine-food-reviews) before running the training script.
