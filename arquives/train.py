import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from utils import load_data

def train():
    mlflow.set_experiment("icecream_sales_model")

    df = load_data("inputs/dataset.csv")

    X = df[["Temperatura"]]
    y = df["Vendas de Sorvete"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)

    with mlflow.start_run():
        mlflow.log_metric("mse", mse)
        mlflow.sklearn.log_model(model, "model")
        print("Modelo treinado e registrado com sucesso.")

if __name__ == "__main__":
    train()