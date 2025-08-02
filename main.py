# rag_engine.py
import os, json, boto3
from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

load_dotenv()

# — S3 & LLM setup —
s3 = boto3.client(
    "s3",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
)
llm = ChatGroq(groq_api_key=os.getenv("GROQ_API_KEY"), model_name="Llama3-8b-8192")

prompt = ChatPromptTemplate.from_template(
    """
    You are a skilled AI. 
    You need to answer the questions only on the basis of the given context and not with your knowledge.
    If the answer to the given question is not present in the context, say "Answer not found in context."
    
    <context>
    {context}
    </context>
    Question: {input}
    """
)

# — Load documents from S3 —
obj = s3.get_object(Bucket="secondbrainv2", Key="processed/texts.json")
texts = json.loads(obj["Body"].read().decode("utf-8"))
documents = [Document(page_content=t) for t in texts]

# — Build vector store & chain once —
embedding      = OllamaEmbeddings(model="gemma:2b")
splitter       = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
split_docs     = splitter.split_documents(documents)
vector_store   = FAISS.from_documents(split_docs, embedding)
retriever      = vector_store.as_retriever()
document_chain = create_stuff_documents_chain(llm, prompt)
retriever_chain = create_retrieval_chain(retriever, document_chain)

# — Expose a simple function —
def ask(question: str) -> str:
    out = retriever_chain.invoke({"input": question})
    return out["answer"]
