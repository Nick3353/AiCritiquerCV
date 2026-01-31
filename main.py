import streamlit as st
import PyPDF2
import io
import os
import cohere
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="AI Resume Critiquer", layout="centered")

st.title("AI Resume Critiquer")
st.markdown("Upload your documents to get info on how to improve it based on your needs.")

COHERE_API_KEY = os.getenv("COHERE_API_KEY")

file_upload = st.file_uploader("Upload your Resume or CV (PDF or text)", type=["pdf", "txt"])
job_role = st.text_input("Which role are you targeting? (Optional)")

analyze = st.button("Analyze Resume")

def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return text

def extract_text_from_file(file_upload):
    if file_upload.type == "application/pdf":
        return extract_text_from_pdf(io.BytesIO(file_upload.read()))
    return file_upload.read().decode("utf-8")

if analyze and file_upload:
    try:
        file_content = extract_text_from_file(file_upload)

        if not file_content.strip():
            st.error("File does not have any content... please retry.")
            st.stop()

        prompt = f"""
Please analyze this resume or CV and provide constructive feedback. if the document sent is not a cv,tell the user to send a cv
Focus on the following aspects:
1. Content clarity and impact
2. Skills presentation
3. Experience description
4. Specific improvements for a role in {job_role if job_role else 'general job or college application'}

Resume content:
{file_content}

Please provide your analysis in a clear, structured format with specific recommendations and examples.
        """

        co = cohere.Client(api_key=COHERE_API_KEY)

        response = co.chat(
            model="command-a-03-2025",
            message=prompt,
            temperature=0.9,
            max_tokens=1000
        )

        st.markdown("### Analysis Results")
        st.markdown(response.text)

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
