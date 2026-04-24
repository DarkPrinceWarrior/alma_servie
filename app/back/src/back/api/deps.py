from pathlib import Path
from typing import Annotated

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from back.core.config import settings
from back.db.database import get_db

DbDep = Annotated[AsyncSession, Depends(get_db)]


def get_data_root() -> Path:
    return settings.data_root


DataRootDep = Annotated[Path, Depends(get_data_root)]
