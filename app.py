import os
import gradio as gr
import pickle

# Загружаем модель
model = pickle.load(open("student_model.pkl", "rb"))

# Функция предсказания
def predict(hours):
    try:
        prediction = model.predict([[float(hours)]])
        return f"Оценка студента: {prediction[0]:.2f}"
    except Exception as e:
        return f"Ошибка: {str(e)}"

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
