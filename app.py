import streamlit as st
import google.generativeai as genai
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import PyPDF2 as pdf
from dotenv import load_dotenv
import json
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re

# Load Environment Variables
load_dotenv()

# Google Gemini Configuration
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

def preprocess_text(text):
    # Lowercase
    text = text.lower()
    # Remove special characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    # Apply stemming
    stemmer = PorterStemmer()
    text = ' '.join([stemmer.stem(word) for word in text.split()])
    return text


# Preprocess Dataset
@st.cache_data
def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()  # Remove whitespace from column names
    df.dropna(inplace=True)  # Drop rows with missing values
    return df

# Train and Evaluate Models
@st.cache_data
def train_and_compare_models(data):
    X = data['Skills_required']  # Input features
    y = data['Job_Role']         # Target variable

    # Vectorize the text data
    vectorizer = TfidfVectorizer(max_features=5000)
    X_vectorized = vectorizer.fit_transform(X)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X_vectorized, y, test_size=0.2, random_state=42
    )

    # Initialize models
    models = {
        "Logistic Regression": LogisticRegression(),
        "Support Vector Machine": SVC(),
        "Random Forest": RandomForestClassifier(),
        "Gradient Boosting": GradientBoostingClassifier(),
        "Naive Bayes": MultinomialNB(),
    }

    model_accuracies = {}
    best_model = None
    best_accuracy = 0

    for model_name, model in models.items():
        # Train the model
        model.fit(X_train, y_train)

        # Predict and calculate accuracy
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        model_accuracies[model_name] = accuracy

        # Track the best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model

    return best_model, vectorizer, model_accuracies

# Extract Text from Uploaded Resume
def extract_pdf_text(uploaded_file):
    reader = pdf.PdfReader(uploaded_file)
    text = ""
    for page in range(len(reader.pages)):
        text += str(reader.pages[page].extract_text())
    return text

# Generate Gemini Response
def get_gemini_response(resume_text, job_description):
    input_prompt = f"""
    Act as an ATS (Applicant Tracking System) for job matching in software and data-related fields.
    Compare the given resume to the job description, score the match percentage, 
    list missing keywords, and provide a brief profile summary.
    Be accurate and structured.
    Resume: {resume_text}
    Job Description: {job_description}

    Response Format:
    {{"JD Match":"%", "MissingKeywords":[], "Profile Summary":""}}"""
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content(input_prompt)
    return response.text

# Streamlit App
st.title("Enhanced ATS Resume Checker")
st.text("Fine-Tuned Resume Screening with Multiple Models & LLM")

# Load Dataset
file_path = "C:\\Users\\yuvar\\Desktop\\Finetuning\\UpdatedResumeDataSet.csv"
st.info("Loading and preprocessing dataset...")
data = load_and_preprocess_data(file_path)

# Train and Compare Models
st.info("Training models and comparing accuracies...")
best_model, vectorizer, model_accuracies = train_and_compare_models(data)

# Display Model Accuracies
st.subheader("Model Accuracies")
for model_name, accuracy in model_accuracies.items():
    st.write(f"{model_name}: {accuracy:.2f}")

# Display Best Model
best_model_name = max(model_accuracies, key=model_accuracies.get)
st.success(f"Best Model: {best_model_name} with accuracy of {model_accuracies[best_model_name]:.2f}")

# User Inputs
job_description = st.text_area("Paste the Job Description")
uploaded_file = st.file_uploader("Upload Your Resume", type=["pdf"])

if st.button("Submit"):
    if uploaded_file is not None and job_description:
        resume_text = extract_pdf_text(uploaded_file)

        # TF-IDF Matching Prediction
        resume_vectorized = vectorizer.transform([resume_text])
        predicted_role = best_model.predict(resume_vectorized)[0]

        # Generate LLM Response
        gemini_response = get_gemini_response(resume_text, job_description)
        parsed_response = json.loads(gemini_response)

        st.subheader("ATS Evaluation Results")
        st.write(f"**Predicted Job Role:** {predicted_role}")
        st.write(f"**Matching Score:** {parsed_response['JD Match']}")
        st.write(f"**Missing Keywords:** {parsed_response['MissingKeywords']}")
        st.write(f"**Profile Summary:** {parsed_response['Profile Summary']}")
    else:
        st.error("Please upload a PDF resume and enter the job description.")
