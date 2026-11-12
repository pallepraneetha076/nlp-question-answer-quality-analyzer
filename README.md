# NLP-Based Question Answer Quality Analyzer

### An Intelligent Student Answer Evaluation System

## Project Overview

The NLP-Based Question Answer Quality Analyzer is a machine learning application designed to evaluate descriptive student answers automatically.

It analyzes student responses based on semantic similarity, question relevance, keyword coverage, answer completeness and text features to estimate marks.

The application provides separate Teacher and Student dashboards using Streamlit.

## Problem Statement

Manual evaluation of descriptive answers can be time-consuming and may involve inconsistencies. This project aims to support the evaluation process through Natural Language Processing and Machine Learning techniques.

## Objectives

- Evaluate student answers automatically.
- Measure semantic similarity between student and reference answers.
- Identify relevant and missing concepts.
- Estimate marks using a trained regression model.
- Provide separate teacher and student interfaces.

## Dataset

**Mohler Short Answer Grading Dataset (MohlerASAG)**

- Source: Hugging Face – `nkazi/MohlerASAG`
- Open-ended records: 2,273
- Unique questions: 79
- Target variable: `score_avg`

The dataset contains questions, instructor reference answers, student answers and human-assigned scores.

## NLP Techniques

1. Text preprocessing and normalization
2. TF-IDF similarity
3. Sentence-BERT semantic embeddings
4. Cosine similarity
5. Reference keyword and concept extraction
6. Sentence-level semantic completeness
7. Answer length analysis

## Machine Learning Algorithms

- Mean Baseline Regression
- Linear Regression
- Ridge Regression
- Random Forest Regression
- Gradient Boosting Regression
- Randomized Search for hyperparameter optimization
- Group-based train-test splitting and GroupKFold cross-validation

**Selected model:** Optimized Gradient Boosting Regressor.

## Model Evaluation

The models are evaluated using:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score

## Application Features

### Teacher Dashboard
- Add new questions.
- Enter reference answers.
- Specify subjects and maximum marks.
- View and manage available questions.

### Student Dashboard
- Select available questions.
- Submit descriptive answers.
- Receive estimated marks.
- View NLP similarity indicators.
- Identify matched and missing concepts.

## Technologies Used

- Python
- Natural Language Processing
- Sentence-BERT
- Scikit-learn
- Pandas and NumPy
- Streamlit
- PyTorch
- Hugging Face

## Project Structure

```text
nlp-question-answer-quality-analyzer/
│
├── app.py
├── student_dashboard.py
├── teacher_dashboard.py
├── model_config.json
├── questions.json
├── question_answer_quality_model.pkl
├── requirements.txt
└── README.md
```

## Installation and Execution

Clone the repository:

```bash
git clone https://github.com/pallepraneetha076/nlp-question-answer-quality-analyzer.git
```

Navigate to the project directory:

```bash
cd nlp-question-answer-quality-analyzer
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

## Limitations

- Predicted marks are estimates and require teacher review.
- The application currently uses a local JSON question bank.
- Adding questions does not automatically retrain the model.
- First-time Sentence-BERT loading may require an internet connection.

## Team Members

| Name | USN |
|---|---|
| Palle Praneetha | 1RVU23BSC096 |
| Monika J | 1RUA24SCS0061 |
| Gajulapalli Sabiha | 1RUA24SCS0035 |
| Gedala Bimala | 1RUA24SCS0037 |
| Archana BS | 1RUA24SCS0011 |

**RV University**  
School of Computer Science & Engineering

---

*This project demonstrates the application of Natural Language Processing and Machine Learning for automated descriptive answer quality assessment.*
