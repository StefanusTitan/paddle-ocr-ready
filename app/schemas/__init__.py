from pydantic import BaseModel
from typing import List, Any, Optional


class PaginationMeta(BaseModel):
    total_items: int
    total_pages: int
    current_page: int
    items_limit: int
    items_count: int


class Pagination(BaseModel):
    items: Optional[List[Any]]
    trips: Optional[List[Any]]
    meta: PaginationMeta
