from typing import Optional

from pydantic import BaseModel


class Passenger(BaseModel):

    Pclass: int
    Sex: str

    Age: Optional[float] = None
    Fare: Optional[float] = None