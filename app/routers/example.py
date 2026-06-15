from fastapi import APIRouter
from app.utils.response import success_response

router = APIRouter()

@router.get("/hello")
async def hello():
    return success_response(message="Hello from Boilerplate!", result={"status": "active"})
