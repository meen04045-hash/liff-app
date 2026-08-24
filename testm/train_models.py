import os
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

# Import the tokenization function from the persistent ai_model module
# so that the pickled objects reference 'ai_model.tokenize' instead of '__main__.tokenize'
from ai_model import tokenize

# =====================================
# TRAIN AND SERIALIZE
# =====================================
def train_and_save(csv_path, model_name):
    print(f"Training {model_name} model from {csv_path}...")
    
    # Load dataset and drop rows with empty values
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["text", "label"])
    
    # Transform text to TF-IDF features
    vectorizer = TfidfVectorizer(
        tokenizer=tokenize,
        token_pattern=None
    )
    X = vectorizer.fit_transform(df["text"])
    y = df["label"]
    
    # Train SVM classifier with probability calibration
    base_model = LinearSVC()
    model = CalibratedClassifierCV(base_model)
    model.fit(X, y)
    
    # Create models output directory
    os.makedirs("models", exist_ok=True)
    
    # Save objects to files
    joblib.dump(vectorizer, f"models/{model_name}_vectorizer.joblib")
    joblib.dump(model, f"models/{model_name}_model.joblib")
    print(f"Saved {model_name} model and vectorizer.")

if __name__ == "__main__":
    datasets = {
        "dataset/risk.csv": "risk",
        "dataset/emotion.csv": "emotion",
        "dataset/problem.csv": "problem",
        "dataset/support_need.csv": "support",
        "dataset/intent.csv": "intent",
        "dataset/conversation_style.csv": "style"
    }
    
    for csv_path, name in datasets.items():
        train_and_save(csv_path, name)
        
    print("All models trained and serialized successfully.")
