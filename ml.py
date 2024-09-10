import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
import pandas as pd
from sklearn.naive_bayes import MultinomialNB

# Load your trained model here
model_expanded = make_pipeline(TfidfVectorizer(), MultinomialNB())
model_expanded.fit(X_train_exp, y_train_exp)

st.title('Chatbot')
user_input = st.text_input('You:', '')

if user_input:
    response = model_expanded.predict([expand_keywords(user_input, synonym_dict)])[0]
    st.write(f'Bot: {response}')


# Load dataset from Excel file
def load_dataset(file_path):
    df = pd.read_excel(file_path)
    return df

# Synonym mapping for procurement-related terms
synonym_dict = {
    "procurement": ["purchase", "acquisition"],
    "tender": ["bid", "proposal"],
    "invoice": ["bill", "receipt"],
    "order": ["purchase", "request"],
    "tracking": ["status", "location"],
    "return": ["refund", "exchange"],
    "policy": ["rule", "regulation"],
    "support": ["help", "assistance"],
    "product": ["item", "goods"],
    "availability": ["in stock", "available"],
    "warranty": ["guarantee", "assurance"]
}

# Function to expand keywords with synonyms
def expand_keywords(keywords, synonym_dict):
    expanded_keywords = set(keywords.split())  # Use a set to avoid duplicates
    for keyword in keywords.split():
        if keyword in synonym_dict:
            expanded_keywords.update(synonym_dict[keyword])
    return ' '.join(expanded_keywords)

# Load the dataset
dataset = load_dataset(r'C:\Users\USER\Desktop\aiassignment\chatbot_dataset.xlsx')

# Apply synonym expansion to the dataset
dataset['Expanded_Keywords'] = dataset['Keywords'].apply(lambda x: expand_keywords(x, synonym_dict))

# Split the data for training and testing
X_expanded = dataset['Expanded_Keywords']
y_expanded = dataset['Response']

X_train_exp, X_test_exp, y_train_exp, y_test_exp = train_test_split(X_expanded, y_expanded, test_size=0.2, random_state=42)

# Create a model pipeline with TF-IDF vectorizer and Naive Bayes classifier
model_expanded = make_pipeline(TfidfVectorizer(), MultinomialNB())

# Train the model with the expanded keywords
model_expanded.fit(X_train_exp, y_train_exp)

# Predict on the test set with expanded keywords
y_pred_expanded = model_expanded.predict(X_test_exp)

# Evaluate the model after synonym expansion
accuracy_expanded = metrics.accuracy_score(y_test_exp, y_pred_expanded)

# Output the accuracy after synonym expansion
print(f'Accuracy after synonym expansion: {accuracy_expanded}')
