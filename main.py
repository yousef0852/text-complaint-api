from sys import prefix
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from interfaces.api.predict_route import router as predict_router
from contextlib import asynccontextmanager
from services.model_loader import ModelLoader
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading model...")
    loader = ModelLoader()
    loader.load_models()
    app.state.model_loader = loader
    yield
    app.state.model_loader = None
    print("Model unloaded")

app = FastAPI(lifespan=lifespan)
app.include_router(
    predict_router,
    tags=["predict"],
)

@app.get("/")
async def root():
    return {"message": "Text Complaint API is online", "status": "healthy"}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


