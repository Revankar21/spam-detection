from flask import Flask, render_template, request, jsonify
from spam_model import SpamDetector
import os

app = Flask(__name__)
detector = SpamDetector()

# Load the pre-trained model if it exists, otherwise train a new one
if os.path.exists('spam_model.joblib') and os.path.exists('vectorizer.joblib'):
    detector.load_model()
else:
    detector.train('spam.csv')
    detector.save_model()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        text = request.form['text']
        if not text:
            return jsonify({'error': 'Please enter some text'})
        
        prediction, probabilities = detector.predict(text)
        
        result = {
            'prediction': 'SPAM' if prediction == 1 else 'HAM',
            'spam_probability': float(probabilities[1]),
            'ham_probability': float(probabilities[0])
        }
        
        return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True) 