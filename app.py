import os
import gradio as gr
import joblib
import pandas as pd

# Загружаем модель
with open("student_model.pkl", "rb") as f:
    data = pickle.load(f)

# Если pickle содержит словарь — достаём модель
model = data['model'] if isinstance(data, dict) and 'model' in data else data

# Функция предсказания
def predict(hours):
    try:
        hours = float(hours)
        df = pd.DataFrame([[hours]], columns=["hours"])
        prediction = model.predict(df)[0]
        return f"Ожидаемый результат студента: {prediction:.2f}"
    except Exception as e:
        return f"Ошибка: {str(e)}"

# Интерфейс Gradio
iface = gr.Interface(
    fn=predict,
    inputs=gr.Number(label="Количество часов обучения"),
    outputs="text",
    title="Student Score Predictor",
    description="Введите количество часов обучения, чтобы предсказать результат студента",
)

# 🔹 Определяем порт из переменной окружения (Render требует именно так)
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    iface.launch(server_name="0.0.0.0", server_port=port)
