import gradio as gr
import pickle
import pandas as pd

# загрузка модели
with open("student_model.pkl", "rb") as f:
    model_data = pickle.load(f)

model = model_data['model']
features = model_data['features']

# функция для перевода текстовых ответов в числа
def encode_input(gender, race, parental_edu, lunch, prep_course, reading_score, writing_score):
    mapping = {
        "gender": {"female": 0, "male": 1},
        "race": {"group A": 0, "group B": 1, "group C": 2, "group D": 3, "group E": 4},
        "parental_edu": {
            "some high school": 0,
            "high school": 1,
            "some college": 2,
            "associate's degree": 3,
            "bachelor's degree": 4,
            "master's degree": 5
        },
        "lunch": {"free/reduced": 0, "standard": 1},
        "prep_course": {"none": 0, "completed": 1}
    }

    data = {
        "gender": mapping["gender"][gender],
        "race/ethnicity": mapping["race"][race],
        "parental level of education": mapping["parental_edu"][parental_edu],
        "lunch": mapping["lunch"][lunch],
        "test preparation course": mapping["prep_course"][prep_course],
        "reading score": reading_score,
        "writing score": writing_score
    }

    return pd.DataFrame([data])

# функция предсказания
def predict_student(gender, race, parental_edu, lunch, prep_course, reading_score, writing_score):
    x = encode_input(gender, race, parental_edu, lunch, prep_course, reading_score, writing_score)
    pred = model.predict(x)[0]
    return "🎓 High (>=70)" if pred == 1 else "📘 Low (<70)"

# интерфейс
iface = gr.Interface(
    fn=predict_student,
    inputs=[
        gr.Radio(["female", "male"], label="Gender"),
        gr.Radio(["group A", "group B", "group C", "group D", "group E"], label="Race/Ethnicity"),
        gr.Radio(
            ["some high school", "high school", "some college", "associate's degree", "bachelor's degree", "master's degree"],
            label="Parental level of education"
        ),
        gr.Radio(["free/reduced", "standard"], label="Lunch type"),
        gr.Radio(["none", "completed"], label="Test preparation course"),
        gr.Slider(0, 100, label="Reading score"),
        gr.Slider(0, 100, label="Writing score")
    ],
    outputs=gr.Label(label="Prediction"),
    title="🎓 Student Performance Predictor",
    description="Определи, получит ли студент высокий балл по математике (High / Low) на основе других показателей."
)

iface.launch()
