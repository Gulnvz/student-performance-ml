import gradio as gr
import pickle

# Загружаем модель
model = pickle.load(open("student_model.pkl", "rb"))

# Функция предсказания
def predict(hours):
    prediction = model.predict([[hours]])
    return f"Оценка студента: {prediction[0]}"

# Интерфейс Gradio
iface = gr.Interface(fn=predict, inputs="number", outputs="text", title="Student Score Predictor")

# Обязательно укажи host='0.0.0.0' и port=7860
iface.launch(server_name="0.0.0.0", server_port=7860)
