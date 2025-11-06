import os
import gradio as gr
import joblib
import pandas as pd

# Загружаем модель
model = joblib.load("student_model.pkl")

# Функция предсказания
def predict(hours):
    try:
        hours = float(hours)
        df = pd.DataFrame([[hours]], columns=["hours"])
        prediction = model.predict(df)[0]
        return f"Ожидаемый результат студента: {prediction:.2f}"
    except Exception as e:
        return f"Ошибка: {str(e)}"

# Создаем интерфейс
iface = gr.Interface(
    fn=predict,
    inputs=gr.Number(label="Количество часов обучения"),
    outputs=gr.Textbox(label="Результат предсказания"),
    title="Student Score Predictor",
    description="Введите количество часов обучения, чтобы предсказать результат студента"
)

# 🔹 Определяем порт из переменной окружения (Render требует именно так)
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    iface.launch(server_name="0.0.0.0", server_port=port)
