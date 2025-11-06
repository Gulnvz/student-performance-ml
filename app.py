import os
import gradio as gr
import joblib
import pandas as pd
import pickle

# Загружаем модель (если она нужна)
model = pickle.load(open("student_model.pkl", "rb"))

# Функция "умного" предсказания
def predict(hours):
    try:
        # Эмуляция предсказания: чем больше часов — тем выше оценка
        score = min(100, round(hours * 1.5 + 20))  # простая формула
        return f"🎓 Оценка студента: {score} баллов"
    except Exception as e:
        return f"Ошибка: {e}"

iface = gr.Interface(
    fn=predict,
    inputs=gr.Number(label="Количество часов обучения", value=5),
    outputs="text",
    title="🎓 Student Score Predictor",
    description="Введите количество часов, чтобы получить прогноз оценки. Чем больше часов — тем выше результат!",
)

# 🔹 Определяем порт из переменной окружения (Render требует именно так)
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    iface.launch(server_name="0.0.0.0", server_port=port)
