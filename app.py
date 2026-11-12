            "Maximum Marks",
            min_value=1.0,
            max_value=100.0,
            value=10.0,
            step=1.0
        )

        submitted = st.form_submit_button("Add Question")

        if submitted:
            if not question.strip() or not reference_answer.strip():
                st.error("Please enter both the question and reference answer.")
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
            with st.expander(f"Q{q['id']}: {q['question']}"):
                st.write("Subject:", q.get("subject", "General"))
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

# ============================================================
# 12. STUDENT DASHBOARD
# ============================================================

elif page == "Student Dashboard":
    st.header("🎓 Student Dashboard")
    questions = load_questions()

    if not questions:
        st.warning("No questions are currently available.")
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
        st.write("Subject:", selected_question.get("subject", "General"))
        st.write("Maximum Marks:", selected_question["max_marks"])

        student_answer = st.text_area(
            "Enter Your Answer",
            height=200,
            placeholder="Write your answer here..."
        )

        if st.button("Evaluate My Answer", type="primary"):
            if not student_answer.strip():
                st.error("Please enter your answer first.")
            else:
                with st.spinner(
                    "Loading the semantic model and analyzing your answer..."
                ):
                    try:
                        result = evaluate_custom_answer(
                            question=selected_question["question"],
                            reference_answer=selected_question["reference_answer"],
                            student_answer=student_answer,
                            max_marks=selected_question["max_marks"]
                        )

                        st.success("Answer evaluation completed!")
                        st.divider()
                        st.subheader("Evaluation Results")

                        col1, col2 = st.columns(2)

                        with col1:
                            st.metric(
                                "Marks Obtained (Estimated)",
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
                        st.write(result["matched_concepts"])

                        st.subheader("Missing Concepts")
                        st.write(result["missing_concepts"])

                        st.warning(
                            "These marks are automated estimates, not final "
                            "teacher-approved grades. A teacher should review "
                            "the answer before assigning final marks."
                        )

                    except Exception as error:
                        st.error(f"Evaluation error: {error}")
                        st.info(
                            "If this is the first evaluation, the Sentence-BERT "
                            "model may need time to download and load."
                        )

