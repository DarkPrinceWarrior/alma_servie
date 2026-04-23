from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    app: str
    version: str
    environment: str


class DbHealthResponse(BaseModel):
    status: str
    database: str
