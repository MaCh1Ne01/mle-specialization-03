from feast import FeatureStore
from feast import (
    Entity,
    FeatureView,
    FileSource,
    Field,
    RequestSource,
    FeatureService,
    PushSource,
)
from feast.types import Int64, Float64, String
from feast.on_demand_feature_view import on_demand_feature_view

companies_stock = Entity(name="companies_stock", join_keys=["companies_stock_id"])

companies_stock_source = FileSource(
    name="companies_stock_source",
    path="data/companies_stock_feature_table.parquet",
    timestamp_field="event_timestamp",
    created_timestamp_column="created",
)


companies_stock_push_source = PushSource(
    name="companies_stock_push_source",
    batch_source=companies_stock_source,
)

companies_stock_view = FeatureView(
    name="companies_stock_view",
    entities=[companies_stock],
    online=True,
    schema=[
        Field(name="Open", dtype=Float64, description="Precio de apertura de la acción."),
        Field(name="High", dtype=Float64, description="Precio más alto de la acción."),
        Field(name="Low", dtype=Float64, description="Precio más bajo de la acción."),
        Field(name="Volume", dtype=Float64, description="Volumen de negociación de las acciones."),
        Field(name="Dividends", dtype=Float64, description="Dividendos pagados por la acción."),
        Field(name="Stock_Splits", dtype=Float64, description="Cualquier split de acciones que haya ocurrido."),
        Field(name="Company", dtype=Float64, description="Símbolo bursátil de la empresa."),
        Field(name="Date_day_of_week", dtype=Float64, description="Día de la semana."),
        Field(name="Date_month", dtype=Float64, description="Mes de la fecha."),
        Field(name="Date_quarter", dtype=Float64, description="Trimestre al que pertenece la fecha."),
        Field(name="Date_is_month_end", dtype=Float64, description="Flag que indica si es fin de mes."),
        Field(name="Date_day_of_year", dtype=Float64, description="Día del año."),
        Field(name="Date_month_sin", dtype=Float64, description="Codificación cíclica del mes (Sin)."),
        Field(name="Date_month_cos", dtype=Float64, description="Codificación cíclica del mes (Cos)."),
        Field(name="Date_day_sin", dtype=Float64, description="Codificación cíclica del día de la semana (Sin)."),
        Field(name="Date_day_cos", dtype=Float64, description="Codificación cíclica del día de la semana (Cos)."),
        Field(name="Close", dtype=Float64, description="Precio de cierre de la acción."),
        Field(name="created", dtype=String, description="Timestamp de creación de features.")
    ],
    source=companies_stock_source,
)

input_request = RequestSource(
    name="input_request",
    schema=[
        Field(name="companies_stock_id", dtype=String),
    ],
)

feature_service = FeatureService(
    name="companies_stock_feature_service",
    features=[companies_stock_view],
)