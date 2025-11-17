# Especialización MLE - Proyecto 03

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Este proyecto corresponde al curso MLE 3 de la Especialización MLE.

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         package_mle_03 and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── package_mle_03   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes package_mle_03 a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    |── utils
    |   └── helpers.py          <- Customized functions
    |
    └── plots.py                <- Code to create visualizations
```

## Problemática
Se requiere implementar un modelo de forecasting para predecir los precios de cierre de acciones de compañías diariamente con el objetivo de planificar la mejor estrategia de Trading que permita obtener la mayor rentabilidad posible de los clientes.

## Diagrama de flujo
<img src="https://drive.google.com/uc?export=view&id=1gX15Ta7DtSOdPLy-vONkdK4m81F4Y6IM" allow="autoplay">

## Descripción del dataset
El conjunto de datos fue extraído del repositorio de datasets de Kaggle, dicho conjunto de datos contiene información histórica (2018-2023) del mercado de valores extraída de Yahoo Finance, incluyendo métricas de rendimiento bursátil de las top 500 empresas. El dataset contiene un total de 602962 registros y 9 características, las cuales se detallan a continuación:

| **Feature**         | **Descripción**                                                                                                        |
|:-------------------:|:----------------------------------------------------------------------------------------------------------------------:|
| Date                | La fecha correspondiente a los datos bursátiles registrados.                                                           |
| Open                | Precio de apertura de la acción en una fecha determinada.                                                              |
| High                | Precio máximo alcanzado por la acción en una fecha determinada.                                                        |
| Low                 | Precio mínimo alcanzado por la acción en una fecha determinada.                                                        |
| Close               | Precio de cierre de la acción en una fecha determinada.                                                                |
| Volume              | Volumen de acciones negociadas en una fecha determinada.                                                               |
| Dividends           | Dividendos pagados por la empresa en una fecha determinada (en caso aplique).                                          |
| Stock Splits        | Cantidad de desdoblamientos de acciones ocurrido en una fecha determinada.                              |
| Company             | Símbolo bursátil que identifica a la empresa a la que pertenece la acción.                                             |

## Model Card
<img src="https://drive.google.com/uc?export=view&id=1fUqmUpDA2wSIrindQZNTXrywLWLjndlb" allow="autoplay">

## Resultados con métricas de evaluación
<img src="https://drive.google.com/uc?export=view&id=1uC02gi9L-U6EgukmwvKOCvPhBl-w0OUu" allow="autoplay">
<img src="https://drive.google.com/uc?export=view&id=1QZsuNd3v61gY0XsliaCAvd8EnSo0Azlo" allow="autoplay">

## Conclusiones
* Los modelos XGBoost y LightGBM presentan prácticamente el mismo rendimiento, ambos con un MAE cercano a 9 y un R² alrededor de 0.93, por lo que se podría inferir que los modelos basados en árboles no son los idóneos para este dataset.

* El modelo ElasticNet de forma individual alcanzó un rendimiento muy alto, el cual compite con el performance del Ridge Stacking Ensemble Regressor (Ridge de predicciones ElasticNet, XGBoost y LightGBM), ambos con un MAE cercano a 1.12 y un R² de 0.9999, lo cual indica que la relación lineal domina el problema. Además, esta similitud entre ambos modelos se corrobora con la feature importance obtenida del modelo Ensemble, dando un 99% de relevancia a las predicciones del modelo lineal individual.

* El modelo Ridge Stacking Ensemble Regressor obtuvo un MAPE por debajo del 1%, lo cual indica que la precisión alcanzada es comercialmente viable, ya que un 0.68% de error porcentual significa que las predicciones son extremadamente confiables para la toma de decisiones, por lo que es el idóneo a llevar a un ambiente productivo.

* El modelo Ensemble captura lo mejor de cada modelo invidual preentrenado, ya que aprovecha la precisión lineal de ElasticNet y corrige los sesgos con los patrones capturados por los modelos basados en árboles XGBoost Regressor y LightGBM Regressor.

* Para el mejor modelo, el Ridge Stacking Ensemble Regressor, se analizaron las feature importances para cada modelo base y se obtuvo que High y Low tienen la mayor relevancia, seguidos de Open. Además, la feature Company no presenta relevancia (PDP horizontal).

## Model Performance Monitoring
<img src="https://drive.google.com/uc?export=view&id=1Na0h19NWhl9jfCqJkloxCvkMSJKA8N9Q" allow="autoplay">

## Manual de ejecución
Revisar las instrucciones en la [carpeta de referencias.](./references)

## Enlaces de experimentos
* DagsHub + MLflow: https://dagshub.com/MaCh1Ne01/mle-specialization-03.mlflow/
* Modelo Productivo: https://dagshub.com/MaCh1Ne01/mle-specialization-03.mlflow/#/models/StockCompaniesForecastingModel