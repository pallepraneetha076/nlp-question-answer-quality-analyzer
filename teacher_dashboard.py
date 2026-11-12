
import streamlit as st
import json
import os

QUESTIONS_FILE = "questions.json"

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

    if not question.strip() or not reference_answer.strip():
        return False

    new_id = max([q["id"] for q in questions], default=0) + 1

    questions.append({
        "id": new_id,
        "question": question.strip(),
        "reference_answer": reference_answer.strip(),
        "max_marks": float(max_marks),
        "subject": subject.strip() or "General"
    })

    save_questions(questions)
    return True

st.set_page_config(
    page_title="Question Answer Quality Analyzer",
    page_icon="📚",
    layout="wide"
)

st.title("NLP-Based Question Answer Quality Analyzer")
st.caption("Intelligent Student Answer Evaluation System")

st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select Dashboard",
    ["Teacher Dashboard", "Student Dashboard"]
)

if page == "Teacher Dashboard":

    st.header("Teacher Dashboard")
    st.subheader("Create a New Question")

    with st.form("add_question_form"):

        question = st.text_area("Enter Question")

        subject = st.text_input(
            "Subject",
            placeholder="e.g. Machine Learning"
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

        submitted = st.form_submit_button("Save Question")

        if submitted:

            if not question.strip() or not reference_answer.strip():
                st.error("Please fill in both question and reference answer.")

            else:
                if add_question(
                    question,
                    reference_answer,
                    max_marks,
                    subject
                ):
                    st.success("Question added successfully!")
                    st.rerun()

    st.divider()

    st.subheader("Available Questions")

    questions = load_questions()

    if questions:

        for q in questions:

            with st.expander(
                f"Question {q['id']}: {q['question']}"
            ):
                st.write("Subject:", q["subject"])
                st.write("Reference Answer:", q["reference_answer"])
                st.write("Maximum Marks:", q["max_marks"])

    else:
        st.info("No questions available.")

else:
    st.header("Student Dashboard")
    st.info("Student evaluation interface will be added next.")
