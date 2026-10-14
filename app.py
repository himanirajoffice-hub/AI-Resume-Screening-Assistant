
import streamlit as st
from dotenv import load_dotenv
import os

load_dotenv()

try:
    groq_key = st.secrets["GROQ_API_KEY"]
    os.environ["GROQ_API_KEY"] = groq_key
except (KeyError, FileNotFoundError):
    pass
import tempfile
import os
import json
import re
import pandas as pd

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

st.set_page_config(
    page_title="AI Resume Screening Assistant",
    page_icon="📄",
    layout="wide"
)

st.title("AI Resume Screening Assistant")
st.write("Resume Screening using LangChain, RAG, FAISS and Ollama")
st.caption("Academic demonstration only. Human review is required for hiring decisions.")

@st.cache_resource
def load_models():
    llm = ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0,
        api_key=os.getenv("GROQ_API_KEY")
    )

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return llm, embeddings
def evaluate_resume(uploaded_file, job_description, llm, embeddings):
    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = os.path.join(temp_dir, "resume.pdf")

        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        loader = PyPDFLoader(file_path)
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )

        chunks = splitter.split_documents(documents)

        if not chunks:
            raise ValueError("No readable text found in the PDF.")

        vectorstore = FAISS.from_documents(chunks, embeddings)

        retriever = vectorstore.as_retriever(
            search_kwargs={"k": min(len(chunks), 20)}
        )

        query = (
            job_description
            + "\nFind candidate name, experience, education, "
            + "skills, projects and relevant qualifications."
        )

        retrieved = retriever.invoke(query)

        context = "\n\n".join(
            doc.page_content for doc in retrieved
        )

        parser = JsonOutputParser()

        prompt = ChatPromptTemplate.from_template("""
You are an AI Resume Screening Assistant.

Evaluate the resume against the job description.

JOB DESCRIPTION:
{job_description}

RETRIEVED RESUME INFORMATION:
{resume_context}

Use only the retrieved resume information for candidate facts.
Do not invent qualifications or experience.
Do not infer personal characteristics.
Treat missing evidence as "Not mentioned", not proof of absence.
Evaluate only job-related skills and qualifications.

Return a valid JSON object containing exactly these keys:

candidate_name: string
candidate_summary: string
match_score: integer between 0 and 100
matching_skills: list of strings
missing_skills: list of strings
strengths: list of strings
weaknesses: list of strings
recommendation: string
justification: string

Use "Recommended", "Consider", or "Not Recommended"
for the recommendation.

The score is an approximate skills-based assessment.
Do not treat it as a validated hiring prediction.

Return JSON only.
""")

        chain = prompt | llm | parser

        result = chain.invoke({
            "job_description": job_description,
            "resume_context": context
        })

        result["resume_file"] = uploaded_file.name
        result["match_score"] = max(
            0, min(100, int(result.get("match_score", 0)))
        )

        return result

def display_list(title, items):
    st.markdown("**" + title + "**")
    if isinstance(items, list):
        for item in items:
            st.write("- " + str(item))
    else:
        st.write(items)

uploaded_files = st.file_uploader(
    "Upload PDF Resumes",
    type=["pdf"],
    accept_multiple_files=True
)

job_description = st.text_area(
    "Enter Job Description",
    height=180,
    placeholder="Enter required skills, experience and qualifications..."
)

if st.button("Evaluate Candidates", type="primary"):

    if not uploaded_files:
        st.warning("Please upload at least one PDF resume.")

    elif not job_description.strip():
        st.warning("Please enter a job description.")

    else:
        try:
            llm, embeddings = load_models()
            results = []

            for uploaded_file in uploaded_files:
                with st.spinner(
                    "Evaluating " + uploaded_file.name + "..."
                ):
                    try:
                        result = evaluate_resume(
                            uploaded_file,
                            job_description,
                            llm,
                            embeddings
                        )
                        results.append(result)

                    except Exception as e:
                        st.error(
                            f"Error processing {uploaded_file.name}: {e}"
                        )

            if results:
                st.session_state["results"] = results

        except Exception as e:
            st.error(f"Application error: {e}")

if "results" in st.session_state:
    results = st.session_state["results"]

    st.header("Candidate Evaluations")

    for result in results:
        with st.expander(
            result.get("candidate_name", "Candidate"),
            expanded=True
        ):
            st.subheader(result.get("candidate_name", "Candidate"))

            st.metric(
                "Match Score",
                str(result.get("match_score", 0)) + "/100"
            )

            st.write("**Candidate Summary**")
            st.write(result.get("candidate_summary", ""))

            display_list(
                "Matching Skills",
                result.get("matching_skills", [])
            )

            display_list(
                "Missing Skills",
                result.get("missing_skills", [])
            )

            display_list(
                "Strengths",
                result.get("strengths", [])
            )

            display_list(
                "Weaknesses",
                result.get("weaknesses", [])
            )

            st.write("**Hiring Recommendation**")
            st.write(result.get("recommendation", ""))

            st.write("**Justification**")
            st.write(result.get("justification", ""))

    if len(results) > 1:
        st.header("Candidate Comparison")

        comparison = pd.DataFrame([
            {
                "Candidate": r.get("candidate_name", ""),
                "Resume": r.get("resume_file", ""),
                "Match Score": r.get("match_score", 0),
                "Recommendation": r.get("recommendation", "")
            }
            for r in results
        ])

        comparison = comparison.sort_values(
            "Match Score",
            ascending=False
        )

        st.dataframe(comparison, use_container_width=True)

        best = comparison.iloc[0]
        st.write(
            "Highest-scoring candidate:",
            best["Candidate"]
        )

        st.caption(
            "Ranking is based on AI-generated estimates and "
            "must be reviewed by a human recruiter."
        )

    st.download_button(
        "Download Evaluation Results (JSON)",
        data=json.dumps(results, indent=2),
        file_name="resume_evaluations.json",
        mime="application/json"
    )
