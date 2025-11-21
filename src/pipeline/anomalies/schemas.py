import pandera.pandas as pa
from pandera.typing import Series, DateTime
from typing import Optional

class SvodSchema(pa.DataFrameModel):
    """
    Schema for the 'svod' sheet containing reference intervals.
    """
    well: Series[str] = pa.Field(alias="Скв", nullable=True)
    cause: Series[str] = pa.Field(alias="ПричОст", nullable=True)
    start_time: Optional[Series[DateTime]] = pa.Field(alias="Время возникновения аномалии", nullable=True)
    end_time: Optional[Series[DateTime]] = pa.Field(alias="Время остановки скважины", nullable=True)

    class Config:
        coerce = True
        strict = False  # Allow other columns like comments

class WellMetricSchema(pa.DataFrameModel):
    """
    Dynamic schema generator for well timeseries data.
    """
    @classmethod
    def create(cls, metric: str) -> pa.DataFrameSchema:
        return pa.DataFrameSchema({
            f"timestamp_{metric}": pa.Column(pa.DateTime, coerce=True, nullable=True),
            metric: pa.Column(pa.Float, coerce=True, nullable=True),
        }, strict=False)
