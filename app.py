import os
import gradio as gr
import joblib
import pandas as pd

# Загружаем модель
model = joblib.load("model.pkl")

# Функция предсказания
def predict(input_data):
    # input_data — строка с числами, разделёнными пробелом
    data = [float(x) for x in input_data.split()]
    df = pd.DataFrame([data])
    prediction = model.predict(df)[0]
    return f"Предсказание модели: {prediction}"

# Создаём интерфейс
iface = gr.Interface(
    fn=predict,
    inputs="text",
    outputs="text",
    title="Модель машинного обучения",
    description="Введите данные для предсказания"
)

# Запуск сервера с указанием порта
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    iface.launch(server_name="0.0.0.0", server_port=port)
