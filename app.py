import os
import gradio as gr
import pickle

# Загружаем модель
data = pickle.load(open("student_model.pkl", "rb"))

# Если внутри словарь — достаём модель
if isinstance(data, dict):
    model = data.get("model")
else:
    model = data

# Проверка на всякий случай
if not hasattr(model, "predict"):
    raise TypeError("Файл student_model.pkl не содержит корректную модель с методом predict")

# Функция предсказания
def predict(hours):
    try:
        features = [[hours, 0, 0, 0, 0, 0, 0]]
        prediction = model.predict(features)[0]

        # Если результат числовой
        if isinstance(prediction, (int, float)):
            score = round(float(prediction), 2)
            if score >= 50:
                color = "green"
                text = f"🎓 Оценка студента: <span style='color:{color};font-weight:bold'>{score}</span> — отлично!"
            else:
                color = "red"
                text = f"⚠️ Оценка студента: <span style='color:{color};font-weight:bold'>{score}</span> — нужно подтянуть знания!"
        else:
            text = f"Предсказание: {prediction}"

        return text

    except Exception as e:
        return f"Ошибка: {e}"

# Интерфейс Gradio
iface = gr.Interface(
    fn=predict,
    inputs=gr.Number(label="Количество часов обучения"),
    outputs=gr.Textbox(label="Результат предсказания"),
    title="Student Score Predictor",
    description="Введите количество часов, чтобы получить предсказание оценки"
)

# 🔹 Определяем порт из переменной окружения (Render требует именно так)
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    iface.launch(server_name="0.0.0.0", server_port=port)
