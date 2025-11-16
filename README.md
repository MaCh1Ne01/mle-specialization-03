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
Se requiere implementar un modelo de forecasting para predecir los precios de acciones de cierre de compañías diariamente con el objetivo de planificar la mejor estrategia de Trading que permita obtener la mayor rentabilidad posible de los clientes.

## Diagrama de flujo
<img src="https://drive.google.com/uc?export=view&id=1gX15Ta7DtSOdPLy-vONkdK4m81F4Y6IM" allow="autoplay">

## Descripción del dataset
El conjunto de datos fue extraído del repositorio de datasets de Kaggle, dicho conjunto de datos contiene información sobre el comportamiento de compra de los clientes, así como su aceptación a campañas anteriores realizadas por el supermarket. El dataset contiene un total de 336006 registros y 29 características, las cuales se detallan a continuación:

| **Feature**         | **Descripción**                                                                                                        |
|:-------------------:|:----------------------------------------------------------------------------------------------------------------------:|
| ID                  | Código UUID del cliente.                                                                                               |
| Year_Birth          | Año de nacimiento del cliente.                                                                                         |
| Education           | Nivel de educación del cliente.                                                                                        |
| Marital_Status      | Estado civil del cliente.                                                                                              |
| Income              | Monto de ingresos anuales del cliente.                                                                                 |
| Kidhome             | Número de niños por casa familiar del cliente.                                                                         |
| Teenhome            | Número de jóvenes por casa familiar del cliente.                                                                       |
| Dt_Customer         | Fecha de registro del cliente en el sistema del supermarket.                                                           |
| Recency             | Días transcurridos desde la última compra del cliente.                                                                 |
| MntWines            | Monto gastado en vino por el cliente el último año.                                                                    |
| MntFruits           | Monto gastado en frutas por el cliente el último año.                                                                  |
| MntMeatProducts     | Monto gastado en carnes por el cliente el último año.                                                                  |
| MntFishProducts     | Monto gastado en pescados por el cliente el último año.                                                                |
| MntSweetProducts    | Monto gastado en dulces por el cliente el último año.                                                                  |
| MntGoldProds        | Monto gastado en joyas de oro por el cliente el último año.                                                            |
| NumDealsPurchases   | Número de compras realizadas por el cliente en el supermarket.                                                         |
| NumWebPurchases     | Número de compras realizadas por el cliente en la web del supermarket.                                                 |
| NumCatalogPurchases | Número de compras realizadas por el cliente por catálogo del supermarket.                                              |
| NumStorePurchases   | Número de compras realizadas por el cliente presencialmente en el supermarket.                                         |
| NumWebVisitsMonth   | Número de visitas mensuales realizadas por el cliente a la web del supermarket.                                        |
| AcceptedCmp3        | Flag binario que indica si el cliente realizó una compra durante la tercera campaña del supermarket.                   |
| AcceptedCmp4        | Flag binario que indica si el cliente realizó una compra durante la cuarta campaña del supermarket.                    |
| AcceptedCmp5        | Flag binario que indica si el cliente realizó una compra durante la quinta campaña del supermarket.                    |
| AcceptedCmp1        | Flag binario que indica si el cliente realizó una compra durante la primera campaña del supermarket.                   |
| AcceptedCmp2        | Flag binario que indica si el cliente realizó una compra durante la segunda campaña del supermarket.                   |
| Complain            | Flag binario que indica si el cliente realizó una queja del supermarket.                                               |
| Z_CostContact       | Costo que representó el contactar con el cliente.                                                                      |
| Z_Revenue           | Ingresos del cliente después de la compra del producto a través de una campaña de marketing.                           |
| Response            | Flag binario que indica si el cliente realizó una compra durante la actual campaña del supermarket.                    |

## Model Card
<img src="https://drive.google.com/uc?export=view&id=1RYgDcOF8v0pK0pZGsQMs9oo3YD1C1nJ1" allow="autoplay">

## Resultados con métricas de evaluación
<img src="https://drive.google.com/uc?export=view&id=11OLpbaqE2pG3rk25S8C7BNb0QktkkP7Z" allow="autoplay">

## Conclusiones
* Los modelos K-Means y OPTICS+K-Means superan significativamente a Gaussian Mixture en todas las métricas de evaluación para este dataset, por lo que se infiere que los datos tienen una estructura que se adapta mejor a métodos basados en distancias que a modelos probabilísticos.

* La detección de outliers de OPTICS no aporta una mejora sustancial en el rendimiento del ensemble, ya que OPTICS+K-Means ofrece una ventaja mínima sobre el modelo K-Means en Silhouette score y Davies-Bouldin score.

* En general, los valores del Silhouette score son relativamente bajos (inferiores a 0.14), lo cual indica que la estructura de clusters no es muy definida (existe overlap).

* El sampling del 5% fue efectivo para realizar una evaluación comparativa, sin embargo, podría estar ocultando patrones en la estructura completa de datos.

* A pesar de que, teóricamente hay más grados de libertad para configurar el ensemble y mejorar su rendimiento, la complejidad computacional adicional de dicho modelo no se justifica por la mejora marginal en los resultados obtenidos.

* Finalmente, se concluye que, debido al buen rendimiento obtenido (comparativa), baja complejidad, eficiencia computacional, interpretabilidad, escalabilidad y madurez del algoritmo, el modelo K-Means es el idóneo a llevar a un ambiente productivo, ya que se puede explotar aún más con un adecuado tunning de hiperparámetros.


## Manual de ejecución
Revisar las instrucciones en la [carpeta de referencias.](./references)

## Enlaces de experimentos
* DagsHub + MLflow: https://dagshub.com/MaCh1Ne01/mle-specialization-02.mlflow/
* Modelo Productivo: https://dagshub.com/MaCh1Ne01/mle-specialization-02.mlflow/#/models/CustomerSegmentationModel