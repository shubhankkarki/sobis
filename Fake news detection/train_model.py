import pandas as pd
import nltk
import string
import pickle

from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset
print("Loading dataset...")
fake = pd.read_csv("dataset/Fake.csv")
true = pd.read_csv("dataset/True.csv")
print("Dataset loaded")

# Add labels
fake["label"] = 0
true["label"] = 1

# Combine datasets
data = pd.concat([fake, true])
data.reset_index(drop=True, inplace=True)

data = data[['text', 'label']]

# Text cleaning function
nltk.download('stopwords')
STOP_WORDS = set(stopwords.words('english'))

def clean_text(text):
    if not isinstance(text, str):
        return ""
    
    # Remove punctuation using a translation table (much faster than a loop)
    text = text.lower().translate(str.maketrans('', '', string.punctuation))
    
    # Use the pre-defined set for O(1) lookup
    words = [w for w in text.split() if w not in STOP_WORDS]
    
    return " ".join(words)
# Apply cleaning
print("Cleaning text...")
data['clean_text'] = data['text'].apply(clean_text)
print("Cleaning completed")

# Split data
X = data["clean_text"]
y = data["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Convert text → numbers
vectorizer = TfidfVectorizer(max_features=5000)
print("Vectorizing text...")
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train model
print("Training model...")
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)
print("Saving model...")

# Check accuracy
pred = model.predict(X_test_vec)
print("Accuracy:", accuracy_score(y_test, pred))

#  SAVE MODEL
print("Saving model...")
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("Model saved successfully!")