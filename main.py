from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import numpy as np

app = FastAPI()


# Заглушка - простая линейная регрессия
class LinearRegressionModel:
    def __init__(self, weights, bias):
        self.weights = np.array(weights)
        self.bias = bias

    def predict(self, input_data):
        return np.dot(input_data, self.weights) + self.bias


# Создаем экземпляр линейной регрессии с произвольными весами и смещением (bias)
linear_regression_model = LinearRegressionModel(weights=[2.0, 3.0], bias=1.0)


# Модель для валидации данных от клиента
class InputData(BaseModel):
    data: list


# Подключаем обработку статических файлов (HTML)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


# Роут для отображения HTML
@app.get("/", response_class=HTMLResponse)
async def read_item(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# Роут для обработки POST-запроса от клиента
@app.post("/predict")
async def make_prediction(data: InputData):
    try:
        # Имитация модели машинного обучения (линейная регрессия)
        prediction = linear_regression_model.predict(data.data)

        # Здесь вы можете также добавить логирование в базу данных
        return {"prediction": prediction.tolist()}
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    # Запустите приложение с помощью uvicorn
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)
