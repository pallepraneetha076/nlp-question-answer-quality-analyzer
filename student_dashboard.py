
st.header("Student Dashboard")

questions = load_questions()

if not questions:

    st.warning("No questions have been added by the teacher.")

else:

    st.subheader("Select a Question")

    question_options = {
        f"Q{q['id']} - {q['question']}": q
        for q in questions
    }

    selected_label = st.selectbox(
        "Choose your question",
        list(question_options.keys())
    )

    selected_question = question_options[selected_label]

    st.info(selected_question["question"])

    st.write("Subject:", selected_question["subject"])

    st.write("Maximum Marks:", selected_question["max_marks"])

    student_answer = st.text_area(
        "Enter Your Answer",
        height=200,
        placeholder="Write your answer here..."
    )

    if st.button("Submit Answer", type="primary"):

        if not student_answer.strip():

            st.error("Please enter an answer before submitting.")

        else:

            st.session_state["submitted_question"] = selected_question
            st.session_state["submitted_answer"] = student_answer

            st.success("Answer submitted successfully!")

            st.info("NLP evaluation engine integration is coming next.")
