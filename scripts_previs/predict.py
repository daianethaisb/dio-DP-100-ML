import mlflow
import pandas as pd

def load_latest_model():
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name("icecream_sales_model")
    if experiment is None:
        raise Exception("Experimento 'icecream_sales_model' não encontrado.")

    runs = client.search_runs(experiment.experiment_id, order_by=["start_time desc"])
    model_uri = f"runs:/{runs[0].info.run_id}/model"
    model = mlflow.sklearn.load_model(model_uri)
    return model

def prever_vendas():
    print("Lendo novos dados para previsão...")
    df = pd.read_csv("inputs/dataset.csv", encoding='latin1')
    X = df[["Temperatura"]]

    model = load_latest_model()
    df["Previsao Vendas"] = model.predict(X)
    print(df[["Data", "Temperatura", "Previsao Vendas"]])

if __name__ == "__main__":
    prever_vendas()