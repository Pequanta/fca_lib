from pydantic import BaseModel


class BitSet(BaseModel):
    set: int