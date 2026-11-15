import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

app = FastAPI(title="AI-Powered Document RAG Assistant")

CHROMA_PATH = "./chroma_db"
UPLOAD_DIR = "./uploaded_docs"
os.makedirs(UPLOAD_DIR, exist_ok=True)

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vector_store = Chroma(
    collection_name="technical_docs",
    embedding_function=embeddings,
    persist_directory=CHROMA_PATH
)

class QueryRequest(BaseModel):
    question: str
    top_k: int = 4

class QueryResponse(BaseModel):
    answer: str
    sources: list[str]

@app.post("/ingest")
async def ingest_document(file: UploadFile = File(...)):
    if not file.filename.endswith((".pdf", ".txt", ".md")):
        raise HTTPException(status_code=400, detail="Only PDF, TXT, and MD files are supported.")
    
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    loader = PyPDFLoader(file_path) if file.filename.endswith(".pdf") else TextLoader(file_path)
    raw_documents = loader.load()

    # Chunking strategy to preserve semantic context across boundaries
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(raw_documents)
    vector_store.add_documents(chunks)

    return {
        "status": "success",
        "filename": file.filename,
        "chunks_indexed": len(chunks)
    }

@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    retriever = vector_store.as_retriever(search_kwargs={"k": request.top_k})
    retrieved_docs = retriever.invoke(request.question)

    if not retrieved_docs:
        return QueryResponse(
            answer="No relevant documentation found to answer this question.",
            sources=[]
        )

    # Strict system prompt to minimize LLM hallucinations
    system_prompt = (
        "You are a technical documentation assistant. Answer the user's question "
        "strictly using the provided context below. If the answer cannot be found "
        "in the context, state clearly that the provided documentation does not "
        "contain the answer. Do not fabricate information.\n\n"
        "Context:\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{question}")
    ])

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)

    context_text = "\n\n---\n\n".join([doc.page_content for doc in retrieved_docs])
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": context_text, "question": request.question})

    sources = list({doc.metadata.get("source", "unknown") for doc in retrieved_docs})
    return QueryResponse(answer=answer, sources=sources)
