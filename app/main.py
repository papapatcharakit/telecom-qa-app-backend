import uvicorn
from fastapi import FastAPI, Request
from pydantic import BaseModel
from utils import QARetriever

app = FastAPI()

# qa_search = QARetriever()

class TextIn(BaseModel):
    text: str
    
class PredictionOut(BaseModel):
    answer: str

@app.get("/")
def home():
    return {"status_health_check": "OK"}

@app.post("/predict", response_model=PredictionOut)
def predict(payload: TextIn):
    answer = qa_search.get_answer(payload.text)
    return {"answer" : answer}

# initializes the QA model and start the uvicorn app
if __name__ == "__main__":
    qa_search = QARetriever()
    uvicorn.run(app, host="0.0.0.0", port=8000)