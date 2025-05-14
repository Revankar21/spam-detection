# Email Spam Detector

A machine learning-based web application that can classify emails as spam or ham (non-spam) using Natural Language Processing and Flask.

## Features

- Modern web interface built with Bootstrap
- Real-time email text analysis
- Probability scores for both spam and ham classifications
- Easy-to-use API endpoint for integration
- Pre-trained model with automatic retraining capability
- Expanded dataset with over 10,000 emails for training

## Installation

1. Clone this repository:
```bash
git clone https://github.com/Revankar21/spam-detection.git
cd spam-detection
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the Flask application:
```bash
python app.py
```

2. Open your web browser and navigate to:
```
http://localhost:5000
```

3. Enter the email text you want to analyze and click "Analyze Text"

## Model Details

The spam detection model uses:
- TF-IDF (Term Frequency-Inverse Document Frequency) for text vectorization
- Multinomial Naive Bayes classifier for text classification
- NLTK for natural language processing and text preprocessing
- Trained on over 10,000 ham and spam emails

## Dataset

The model is trained on a combination of real and synthetic email data:
- Original SMS Spam Collection dataset
- Additional synthetic data generated to improve model performance
- Total of ~10,500 emails with ~3,200 spam and ~7,300 ham

## API Usage

You can also use the spam detector via API:

```bash
curl -X POST -F "text=your email text here" http://localhost:5000/predict
```

Response format:
```json
{
    "prediction": "SPAM" or "HAM",
    "spam_probability": float,
    "ham_probability": float
}
```

## Directory Structure

```
spam-detection/
├── app.py              # Flask application
├── spam_model.py       # Spam detection model
├── generate_data.py    # Script to generate synthetic training data 
├── requirements.txt    # Python dependencies
├── templates/         
│   └── index.html     # Web interface
├── spam.csv           # Dataset (original + synthetic)
├── spam_model.joblib   # Trained model (generated after first run)
└── vectorizer.joblib   # TF-IDF vectorizer (generated after first run)
```

## Requirements

- Python 3.7+
- Flask
- scikit-learn
- pandas
- numpy
- nltk
- joblib

## License

This project is licensed under the MIT License - see the LICENSE file for details. 