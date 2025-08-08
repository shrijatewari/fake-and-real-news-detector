import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib

# If running for the first time:
nltk.download('stopwords')

# Load the datasets
fake_df = pd.read_csv("/Users/shrijatewari/Downloads/archive/Fake.csv")
real_df = pd.read_csv("/Users/shrijatewari/Downloads/archive/True.csv")

# Label the data
fake_df['label'] = 0  # Fake news
real_df['label'] = 1  # Real news

# Combine datasets
df = pd.concat([fake_df, real_df])
df = df.sample(frac=1).reset_index(drop=True)

# Cleaning function
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = str(text).lower()
    text = "".join([char for char in text if char not in string.punctuation])
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

df["clean_text"] = df["text"].apply(clean_text)

# Split data
X = df["clean_text"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# TF-IDF vectorizer
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train model
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# Evaluate
y_pred = model.predict(X_test_vec)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ✅ Save the trained model and vectorizer
joblib.dump(model, 'fake_news_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')  # ✅ FIXED
