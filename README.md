# Ice Cream Sales Prediction

Este projeto usa Machine Learning para prever vendas de sorvete com base na temperatura ambiente.

## Objetivo

Treinar um modelo preditivo de regressão para prever vendas de sorvete usando regressão linear simples, com rastreamento de experimentos via MLflow.

## Estrutura

- `inputs/dataset.csv`: Dados de exemplo
- `arquives/`: Scripts de treinamento e predição
- `notebooks/`: EDA e modelo

## 🧪 MLflow

Como rodar:

1. Instale as dependências: `pip install -r requirements.txt`
2. Treine o modelo: `python arquives/train.py`
3. Rode o MLflow local: `mlflow ui`
4. Faça previsões: `python scripts_previs/predict.py`

Coloque seu dataset em `inputs/dataset.csv` com colunas:

- Data
- Vendas de Sorvete
- Temperatura (°C)

E visualize os experimentos em `http://localhost:5000`
