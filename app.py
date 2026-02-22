import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2

# Function to extract text from PDF
def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

st.set_page_config(page_title="AI Resume Ranker", page_icon="📄")
st.title("Smart Resume Screener (ATS)")

# Input Job Description
jd_text = st.text_area("Paste the Job Description here:", height=200)

# Upload Resumes
uploaded_files = st.file_uploader("Upload Resumes (PDF format)", type=["pdf"], accept_multiple_files=True)

if st.button("Rank Resumes") and jd_text and uploaded_files:
    with st.spinner('Analyzing resumes...'):
        resume_texts = [extract_text_from_pdf(f) for f in uploaded_files]
        file_names = [f.name for f in uploaded_files]

        all_content = [jd_text] + resume_texts
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(all_content)
        
        scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

        results = sorted(zip(file_names, scores), key=lambda x: x[1], reverse=True)
        
        st.success("Ranking Complete!")
        for name, score in results:
            st.write(f"**{name}**: {round(score * 100, 2)}% Match")
            st.progress(float(score))