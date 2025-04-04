import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import re

nltk.download('stopwords')
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

# Function to clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\W+', ' ', text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in stop_words]
    return ' '.join(tokens)

# Streamlit UI
st.set_page_config(page_title="Resume Analyzer & Job Matcher", layout="centered")
st.title("📄 AI Resume Analyzer & Job Matcher")

st.write("Paste your **Resume** and a **Job Description** below. The system will analyze them and suggest improvements.")

resume_input = st.text_area("📝 Paste Your Resume Here", height=200)
job_input = st.text_area("💼 Paste Job Description Here", height=200)

if st.button("Analyze"):
    if resume_input.strip() == "" or job_input.strip() == "":
        st.warning("Please enter both resume and job description.")
    else:
        # Clean text
        resume_clean = clean_text(resume_input)
        job_clean = clean_text(job_input)

        # TF-IDF Vectorization
        vectorizer = TfidfVectorizer()
        tfidf = vectorizer.fit_transform([resume_clean, job_clean])

        # Cosine Similarity
        score = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
        score_percent = round(score * 100, 2)

        # Extract keywords
        resume_words = set(resume_clean.split())
        job_words = set(job_clean.split())

        matched_keywords = resume_words.intersection(job_words)
        missing_keywords = job_words - resume_words

        # Display results
        st.subheader("📊 Results")
        st.success(f"Match Score: **{score_percent}%**")
        st.write("### ✅ Matched Keywords:")
        st.write(", ".join(matched_keywords) if matched_keywords else "None")

        st.write("### 🔍 Suggested Keywords to Add:")
        st.write(", ".join(missing_keywords) if missing_keywords else "None")
