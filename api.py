from fastapi import FastAPI,HTTPException
from pydantic import BaseModel
from main import retriever_chain

app = FastAPI()

class QueryRequest(BaseModel):
    question:str

@app.post('/ask')
async def ask(req:QueryRequest):
    try:
        resp = retriever_chain.invoke({'input':req.question})
        return {"answer":resp['answer']}
    except Exception as e:
        raise HTTPException(status_code=500,detail=str(e))