# pip install langchain langchain-core langchain-community fastapi uvicorn langchain-ollama gradio faiss-cpu pypdf
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from langserve import add_routes
import gradio as gr
import uvicorn
import os
import tempfile
import shutil

# 임시 디렉토리 설정
UPLOAD_DIR = "uploaded_pdfs"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# 벡터 스토어 디렉토리
VECTOR_STORE_DIR = "faiss_index"

# Ollama 모델 설정
llm = OllamaLLM(
    base_url="http://localhost:11434",
    model="hf.co/QuantFactory/llama-3.2-Korean-Bllossom-3B-GGUF"  # 사용할 모델명
)

# 한국어 임베딩 모델 설정 (kobert 또는 다른 한국어 모델 사용)
embeddings = HuggingFaceEmbeddings(
    model_name="jhgan/ko-sbert-nli",  # 한국어 지원 임베딩 모델
    model_kwargs={'device': 'cpu'}
)

# 벡터 DB 초기화 변수
vector_store = None

# 문서 로드 및 벡터 스토어 생성 함수
def process_pdf_documents(file_paths):
    # 문서 로드
    documents = []
    for file_path in file_paths:
        loader = PyPDFLoader(file_path)
        documents.extend(loader.load())
    
    # 문서 분할
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    chunks = text_splitter.split_documents(documents)
    
    # FAISS 벡터 스토어 생성
    vector_store = FAISS.from_documents(chunks, embeddings)
    
    # 벡터 스토어 저장
    vector_store.save_local(VECTOR_STORE_DIR, allow_dangerous_deserialization=True)
    
    return vector_store

# 기존 벡터 스토어 로드 함수
def load_vector_store():
    if os.path.exists(VECTOR_STORE_DIR):
        return FAISS.load_local(VECTOR_STORE_DIR, embeddings, allow_dangerous_deserialization=True)
    return None

# 벡터 스토어 로드 시도
vector_store = load_vector_store()

# RAG를 위한 프롬프트 템플릿
rag_prompt_template = """
당신은 주어진 문서를 바탕으로 정확한 정보를 제공하는 도우미입니다.
다음 정보를 참고하여 질문에 답변해주세요:

{context}

질문: {query}
답변:
"""

rag_prompt = PromptTemplate(
    input_variables=["context", "query"],
    template=rag_prompt_template
)

# RAG 체인 생성 함수
def create_rag_chain():
    if vector_store is None:
        return None
    
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )
    
    # RetrievalQA 체인 생성
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": rag_prompt}
    )
    
    return qa_chain

# FastAPI 앱 생성
app = FastAPI(
    title="LangChain + Ollama + FAISS RAG API",
    description="PDF 문서를 기반으로 질문 답변하는 RAG 시스템",
)

# 파일 업로드 엔드포인트
@app.post("/upload-pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    global vector_store
    
    # 파일 저장
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # 벡터 스토어 처리
    file_paths = [file_path]
    if vector_store:
        # 기존 벡터 스토어에 문서 추가
        new_docs = PyPDFLoader(file_path).load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        chunks = text_splitter.split_documents(new_docs)
        vector_store.add_documents(chunks)
        vector_store.save_local(VECTOR_STORE_DIR)
    else:
        # 새 벡터 스토어 생성
        vector_store = process_pdf_documents(file_paths)
    
    return {"message": f"PDF 파일 '{file.filename}'이 성공적으로 업로드되고 처리되었습니다."}

# 질문 처리 엔드포인트
@app.post("/question/")
async def ask_question(query: str = Form(...)):
    if vector_store is None:
        return JSONResponse(
            status_code=400,
            content={"error": "먼저 PDF 문서를 업로드해주세요."}
        )

    rag_chain = create_rag_chain()
    if rag_chain is None:
        return JSONResponse(
            status_code=500,
            content={"error": "RAG 체인을 생성할 수 없습니다."}
        )

    # Pass a dictionary with 'query' key
    response = rag_chain.invoke({"query": query})

    # 결과에서 answer 키 추출
    answer = response.get("result", "응답을 생성할 수 없습니다.")

    return {"answer": answer}

# 기본 경로
@app.get("/")
async def root():
    return {"message": "RAG + Ollama 서비스가 실행 중입니다. PDF를 업로드하고 질문을 해보세요."}

# LangServe 라우트 추가 (기본 체인)
simple_chain = PromptTemplate(
    input_variables=["query"],
    template="다음 질문에 대해 답변해주세요: {query}"
) | llm

add_routes(
    app,
    simple_chain,
    path="/ollama-chain"
)

# Gradio 인터페이스 생성 함수
def create_gradio_interface():
    def upload_pdf_file(file):
        if file is None:
            return "파일을 선택해주세요."
            
        file_path = file.name
        temp_path = os.path.join(UPLOAD_DIR, os.path.basename(file_path))
        
        # 파일 복사
        shutil.copy(file_path, temp_path)
        
        global vector_store
        if vector_store:
            # 기존 벡터 스토어에 문서 추가
            new_docs = PyPDFLoader(temp_path).load()
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
            )
            chunks = text_splitter.split_documents(new_docs)
            vector_store.add_documents(chunks)
            vector_store.save_local(VECTOR_STORE_DIR)
        else:
            # 새 벡터 스토어 생성
            vector_store = process_pdf_documents([temp_path])
        
        return f"PDF 파일 '{os.path.basename(file_path)}'이 성공적으로 업로드되고 처리되었습니다."
    
    def query_rag(question):
        global vector_store
        if vector_store is None:
            return "먼저 PDF 문서를 업로드해주세요."

        if not question or question.strip() == "":
            return "질문을 입력해주세요."

        rag_chain = create_rag_chain()
        if rag_chain is None:
            return "RAG 체인을 생성할 수 없습니다."

        # Change this line to pass a dictionary with 'query' key
        response = rag_chain.invoke({"query": question})

        # 결과에서 answer 키 추출
        return response.get("result", "응답을 생성할 수 없습니다.")
    
    with gr.Blocks(theme=gr.themes.Soft()) as interface:
        gr.Markdown("# 한국어 RAG 문서 질의응답 시스템")
        gr.Markdown("PDF 문서를 업로드하고 질문하면 문서 내용을 바탕으로 답변해드립니다.")
        
        with gr.Tab("PDF 업로드"):
            file_input = gr.File(label="PDF 파일 업로드")
            upload_button = gr.Button("업로드 및 처리")
            upload_output = gr.Textbox(label="업로드 결과")
            
            upload_button.click(
                fn=upload_pdf_file,
                inputs=file_input,
                outputs=upload_output
            )
        
        with gr.Tab("질문하기"):
            question_input = gr.Textbox(lines=3, label="질문을 입력하세요")
            submit_button = gr.Button("질문하기")
            answer_output = gr.Textbox(label="AI 답변")
            
            submit_button.click(
                fn=query_rag,
                inputs=question_input,
                outputs=answer_output
            )
    
    return interface

# Gradio 앱 생성
gradio_app = create_gradio_interface()

# Gradio 앱을 FastAPI에 마운트
app = gr.mount_gradio_app(app, gradio_app, path="/ui")

# 서버 실행 코드
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)