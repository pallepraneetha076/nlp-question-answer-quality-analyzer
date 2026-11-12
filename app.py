
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
import re
import html

from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ------------------------------------------
# 1. PAGE CONFIGURATION
# ------------------------------------------

st.set_page_config(
    page_title="NLP-Based Question Answer Quality Analyzer",
    page_icon="📚",
    layout="wide"
)

# ------------------------------------------
# 2. LOAD TRAINED MODEL
# ------------------------------------------

@st.cache_resource
def load_ml_model():

    model = joblib.load("question_answer_quality_model.pkl")

    embedding_model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    return model, embedding_model


model, sbert_model = load_ml_model()

# ------------------------------------------
# 3. MODEL CONFIGURATION
# ------------------------------------------

with open("model_config.json", "r") as file:
    model_config = json.load(file)

feature_columns = model_config["feature_columns"]

QUESTIONS_FILE = "questions.json"

# ------------------------------------------
# 4. QUESTION DATABASE FUNCTIONS
# ------------------------------------------

def load_questions():

    if not os.path.exists(QUESTIONS_FILE):
        return []

    with open(QUESTIONS_FILE, "r") as file:
        return json.load(file)


def save_questions(questions):

    with open(QUESTIONS_FILE, "w") as file:
        json.dump(questions, file, indent=4)


def add_question(question, reference_answer, max_marks, subject):

    questions = load_questions()

    new_id = max(
        [q["id"] for q in questions],
        default=0
    ) + 1

    new_question = {
        "id": new_id,
        "question": question.strip(),
        "reference_answer": reference_answer.strip(),
        "max_marks": float(max_marks),
        "subject": subject.strip() or "General"
    }

    questions.append(new_question)
    save_questions(questions)


def delete_question(question_id):

    questions = load_questions()

    updated_questions = [
        q for q in questions
        if q["id"] != question_id
    ]

    save_questions(updated_questions)

# ------------------------------------------
# 5. TEXT PREPROCESSING
# ------------------------------------------

def preprocess_custom_text(text):

    text = str(text)
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ------------------------------------------
# 6. TF-IDF SIMILARITY
# ------------------------------------------

def custom_tfidf_similarity(text1, text2):

    try:

        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            stop_words="english"
        )

        vectors = vectorizer.fit_transform([text1, text2])

        return float(
            cosine_similarity(
                vectors[0],
                vectors[1]
            )[0][0]
        )

    except ValueError:

        return 0.0


# ------------------------------------------
# 7. REFERENCE KEYWORD COVERAGE
# ------------------------------------------

def custom_concept_coverage(reference_answer, student_answer):

    try:

        vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_features=8
        )

        vectorizer.fit_transform([reference_answer])

        reference_concepts = (
            vectorizer.get_feature_names_out().tolist()
        )

    except ValueError:

        reference_concepts = []

    student_words = re.findall(
        r"\b[a-zA-Z][a-zA-Z0-9+#.-]*\b",
        student_answer.lower()
    )

    student_concepts = set(student_words)

    student_concepts.update(
        " ".join(student_words[i:i+2])
        for i in range(len(student_words)-1)
    )

    matched = [
        term for term in reference_concepts
        if term in student_concepts
    ]

    missing = [
        term for term in reference_concepts
        if term not in student_concepts
    ]

    coverage = (
        len(matched) / len(reference_concepts)
        if reference_concepts else 0.0
    )

    return coverage, matched, missing


# ------------------------------------------
# 8. SEMANTIC COMPLETENESS
# ------------------------------------------

def custom_semantic_completeness(reference_answer, student_answer):

    reference_sentences = [
        s.strip()
        for s in re.split(
            r'(?<=[.!?])\s+',
            reference_answer.strip()
        )
        if s.strip()
    ]

    student_sentences = [
        s.strip()
        for s in re.split(
            r'(?<=[.!?])\s+',
            student_answer.strip()
        )
        if s.strip()
    ]

    if not reference_sentences or not student_sentences:
        return 0.0

    reference_vectors = sbert_model.encode(
        reference_sentences,
        normalize_embeddings=True
    )

    student_vectors = sbert_model.encode(
        student_sentences,
        normalize_embeddings=True
    )

    similarity_matrix = cosine_similarity(
        reference_vectors,
        student_vectors
    )

    best_matches = np.max(
        similarity_matrix,
        axis=1
    )

    return float(
        np.mean(np.clip(best_matches, 0, 1))
    )


# ------------------------------------------
# 9. CUSTOM ANSWER EVALUATION ENGINE
# ------------------------------------------

def evaluate_custom_answer(
    question,
    reference_answer,
    student_answer,
    max_marks=5
):

    question = preprocess_custom_text(question)
    reference_answer = preprocess_custom_text(reference_answer)
    student_answer = preprocess_custom_text(student_answer)

    if not question or not reference_answer or not student_answer:
        raise ValueError("All answer evaluation fields are required.")

    # TF-IDF features
    tfidf_similarity = custom_tfidf_similarity(
        reference_answer,
        student_answer
    )

    question_tfidf_relevance = custom_tfidf_similarity(
        question,
        student_answer
    )

    # SBERT embeddings
    student_vector = sbert_model.encode(
        student_answer,
        normalize_embeddings=True
    )

    reference_vector = sbert_model.encode(
        reference_answer,
        normalize_embeddings=True
    )

    question_vector = sbert_model.encode(
        question,
        normalize_embeddings=True
    )

    sbert_similarity = float(
        np.dot(student_vector, reference_vector)
    )

    sbert_question_relevance = float(
        np.dot(student_vector, question_vector)
    )

    # Keyword coverage
    keyword_coverage, matched, missing = (
        custom_concept_coverage(
            reference_answer,
            student_answer
        )
    )

    # Semantic completeness
    semantic_completeness = custom_semantic_completeness(
        reference_answer,
        student_answer
    )

    # Length features
    student_word_count = len(student_answer.split())
    reference_word_count = len(reference_answer.split())
    question_word_count = len(question.split())

    answer_reference_length_ratio = min(
        student_word_count / max(reference_word_count, 1),
        5
    )

    # Model features
    features = {
        "tfidf_similarity": tfidf_similarity,
        "question_tfidf_relevance": question_tfidf_relevance,
        "sbert_similarity": sbert_similarity,
        "sbert_question_relevance": sbert_question_relevance,
        "reference_keyword_coverage": keyword_coverage,
        "semantic_completeness": semantic_completeness,
        "student_word_count": student_word_count,
        "reference_word_count": reference_word_count,
        "answer_reference_length_ratio": answer_reference_length_ratio,
        "question_word_count": question_word_count
    }

    feature_input = pd.DataFrame(
        [[features[col] for col in feature_columns]],
        columns=feature_columns
    )

    predicted_score = float(
        model.predict(feature_input)[0]
    )

    predicted_score = float(
        np.clip(predicted_score, 0, 5)
    )

    final_marks = (
        predicted_score / 5
    ) * float(max_marks)

    return {
        "predicted_score": predicted_score,
        "final_marks": final_marks,
        "maximum_marks": float(max_marks),
        "tfidf_similarity": tfidf_similarity,
        "sbert_similarity": sbert_similarity,
        "question_relevance": sbert_question_relevance,
        "keyword_coverage": keyword_coverage,
        "semantic_completeness": semantic_completeness,
        "matched_concepts": matched,
        "missing_concepts": missing
    }


# ------------------------------------------
# 10. APPLICATION HEADER
# ------------------------------------------

st.title("📚 NLP-Based Question Answer Quality Analyzer")

st.markdown(
    "### Intelligent Student Answer Evaluation System"
)

st.caption(
    "Machine Learning | Natural Language Processing | "
    "Semantic Analysis"
)

st.divider()

# ------------------------------------------
# 11. SIDEBAR NAVIGATION
# ------------------------------------------

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Choose Dashboard",
    [
        "Home",
        "Teacher Dashboard",
        "Student Dashboard"
    ]
)

# ------------------------------------------
# 12. HOME PAGE
# ------------------------------------------

if page == "Home":

    st.header("Welcome to the Answer Quality Analyzer")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("👩‍🏫 Teacher Portal")
        st.write(
            "Create questions, provide reference answers "
            "and set maximum marks."
        )

    with col2:
        st.subheader("🎓 Student Portal")
        st.write(
            "Answer questions and receive automated "
            "NLP-based evaluation."
        )

    questions = load_questions()

    st.metric(
        "Available Questions",
        len(questions)
    )

    st.info(
        "Select a dashboard from the sidebar to get started."
    )


# ------------------------------------------
# 13. TEACHER DASHBOARD
# ------------------------------------------

elif page == "Teacher Dashboard":

    st.header("👩‍🏫 Teacher Dashboard")

    st.subheader("Add a New Question")

    with st.form("teacher_question_form"):

        question = st.text_area(
            "Enter Question"
        )

        subject = st.text_input(
            "Subject",
            placeholder="Machine Learning"
        )

        reference_answer = st.text_area(
            "Enter Reference Answer",
            height=150
        )

        max_marks = st.number_input(
            "Maximum Marks",
            min_value=1.0,
            max_value=100.0,
            value=10.0,
            step=1.0
        )

        submitted = st.form_submit_button(
            "Add Question"
        )

        if submitted:

            if not question.strip() or not reference_answer.strip():

                st.error(
                    "Please enter both the question and reference answer."
                )

            else:

                add_question(
                    question,
                    reference_answer,
                    max_marks,
                    subject
                )

                st.success("Question added successfully!")
                st.rerun()

    st.divider()

    st.subheader("Manage Existing Questions")

    questions = load_questions()

    if questions:

        for q in questions:

            with st.expander(
                f"Q{q['id']}: {q['question']}"
            ):

                st.write("Subject:", q["subject"])
                st.write("Reference Answer:", q["reference_answer"])
                st.write("Maximum Marks:", q["max_marks"])

                if st.button(
                    "Delete Question",
                    key=f"delete_{q['id']}"
                ):

                    delete_question(q["id"])
                    st.success("Question deleted.")
                    st.rerun()

    else:

        st.info("No questions available.")


# ------------------------------------------
# 14. STUDENT DASHBOARD
# ------------------------------------------

elif page == "Student Dashboard":

    st.header("🎓 Student Dashboard")

    questions = load_questions()

    if not questions:

        st.warning(
            "No questions are currently available."
        )

    else:

        question_options = {
            f"Q{q['id']} - {q['question']}": q
            for q in questions
        }

        selected_label = st.selectbox(
            "Select a Question",
            list(question_options.keys())
        )

        selected_question = question_options[selected_label]

        st.info(selected_question["question"])

        st.write("Subject:", selected_question["subject"])

        st.write(
            "Maximum Marks:",
            selected_question["max_marks"]
        )

        student_answer = st.text_area(
            "Enter Your Answer",
            height=200,
            placeholder="Write your answer here..."
        )

        if st.button(
            "Evaluate My Answer",
            type="primary"
        ):

            if not student_answer.strip():

                st.error(
                    "Please enter your answer first."
                )

            else:

                with st.spinner(
                    "Analyzing your answer using NLP..."
                ):

                    try:

                        result = evaluate_custom_answer(
                            question=selected_question["question"],
                            reference_answer=selected_question["reference_answer"],
                            student_answer=student_answer,
                            max_marks=selected_question["max_marks"]
                        )

                        st.success(
                            "Answer evaluation completed!"
                        )

                        st.divider()

                        st.subheader("Evaluation Results")

                        col1, col2 = st.columns(2)

                        with col1:

                            st.metric(
                                "Predicted Marks",
                                f"{result['final_marks']:.2f} / "
                                f"{result['maximum_marks']:.2f}"
                            )

                        with col2:

                            st.metric(
                                "Semantic Similarity",
                                f"{result['sbert_similarity']:.3f}"
                            )

                        st.subheader("NLP Analysis")

                        metrics = pd.DataFrame({
                            "Metric": [
                                "TF-IDF Similarity",
                                "Question Relevance",
                                "Keyword Coverage",
                                "Semantic Completeness"
                            ],
                            "Score": [
                                result["tfidf_similarity"],
                                result["question_relevance"],
                                result["keyword_coverage"],
                                result["semantic_completeness"]
                            ]
                        })

                        st.dataframe(
                            metrics,
                            use_container_width=True,
                            hide_index=True
                        )

                        st.subheader("Matched Concepts")

                        st.write(
                            result["matched_concepts"]
                        )

                        st.subheader("Missing Concepts")

                        st.write(
                            result["missing_concepts"]
                        )

                        st.warning(
                            "These marks are automated predictions, "
                            "not final teacher-approved grades. "
                            "Missing keywords do not necessarily mean "
                            "a concept is absent."
                        )

                    except Exception as error:

                        st.error(
                            f"Evaluation error: {error}"
                        )
