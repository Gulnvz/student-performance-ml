import os
import gradio as gr
import joblib
import pandas as pd

# Загружаем модель
model = joblib.load("student_model.pkl")

def predict_performance(gender, race, parental_education, lunch, prep_course, reading, writing):
    data = pd.DataFrame([[gender, race, parental_education, lunch, prep_course, reading, writing]],
                        columns=["gender", "race/ethnicity", "parental level of education", "lunch", "test preparation course", "reading score", "writing score"])
    prediction = model.predict(data)[0]
    return f"🎓 Предсказание уровня успеваемости: {prediction}"

iface = gr.Interface(
    fn=predict_performance,
    inputs=[
        gr.Dropdown(["male", "female"], label="Gender"),
        gr.Dropdown(["group A", "group B", "group C", "group D", "group E"], label="Race/Ethnicity"),
        gr.Textbox(label="Parental Level of Education"),
        gr.Dropdown(["standard", "free/reduced"], label="Lunch"),
        gr.Dropdown(["none", "completed"], label="Test Preparation Course"),
        gr.Number(label="Reading Score"),
        gr.Number(label="Writing Score"),
    ],
    outputs="text",
    title="Student Performance Predictor",
    description="Введите данные, чтобы предсказать уровень успеваемости ученика."
)

# 🔹 Обновлённый запуск для Render
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    # Render требует listen на 0.0.0.0 и без share
    iface.launch(server_name="0.0.0.0", server_port=port, share=False, inbrowser=False, quiet=True)
