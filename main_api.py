from fastapi import FastAPI
from api import endpoints

app = FastAPI(title='Respira Mais API')
app.include_router(endpoints.router)
