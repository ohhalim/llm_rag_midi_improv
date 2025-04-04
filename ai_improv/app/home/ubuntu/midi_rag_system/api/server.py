"""
FastAPI 웹 서버 모듈
MIDI RAG 시스템의 API 엔드포인트를 구현합니다.
"""
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import glob
import asyncio
from typing import List, Optional
import shutil
from pydantic import BaseModel

import config
from midi_rag_system.core.midi_rag_system import MIDIRAGSystem

# FastAPI 앱 초기화
app = FastAPI(
    title="MIDI RAG API",
    description="MCP, LangChain, YuE를 활용한 MIDI 기반 AI 즉흥 연주 시스템 API",
    version="1.0.0"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MIDI RAG 시스템 초기화
midi_rag_system = MIDIRAGSystem()

# 벡터 저장소 로드 시도
try:
    if os.path.exists(config.VECTORSTORE_PATH):
        midi_rag_system.load_vectorstore(config.VECTORSTORE_PATH)
        print(f"벡터 저장소 로드 완료: {config.VECTORSTORE_PATH}")
except Exception as e:
    print(f"벡터 저장소 로드 실패: {str(e)}")

# 요청 모델 정의
class GenerateRequest(BaseModel):
    genre: Optional[str] = None
    lyrics: Optional[str] = None
    output_format: str = "midi"

@app.get("/")
async def root():
    """API 상태 확인"""
    return {"status": "online", "message": "MIDI RAG API가 실행 중입니다."}

@app.post("/train")
async def train_system(background_tasks: BackgroundTasks):
    """
    학습용 MIDI 파일로 시스템 학습
    
    학습 디렉토리의 모든 MIDI 파일을 사용하여 벡터 저장소를 생성합니다.
    """
    # 학습 디렉토리에서 MIDI 파일 목록 가져오기
    midi_files = glob.glob(str(config.TRAINING_DIR / "*.mid")) + glob.glob(str(config.TRAINING_DIR / "*.midi"))
    
    if not midi_files:
        raise HTTPException(status_code=404, detail="학습 디렉토리에 MIDI 파일이 없습니다.")
    
    # 백그라운드에서 학습 실행
    def train_task():
        midi_rag_system.train(midi_files, config.VECTORSTORE_PATH)
    
    background_tasks.add_task(train_task)
    
    return {
        "status": "training",
        "message": f"학습이 시작되었습니다. {len(midi_files)}개의 MIDI 파일을 처리합니다.",
        "files": midi_files
    }

@app.post("/upload/training")
async def upload_training_file(file: UploadFile = File(...)):
    """
    학습용 MIDI 파일 업로드
    
    업로드된 파일은 학습 디렉토리에 저장됩니다.
    """
    # 파일 확장자 확인
    if not file.filename.lower().endswith(('.mid', '.midi')):
        raise HTTPException(status_code=400, detail="MIDI 파일만 업로드 가능합니다.")
    
    # 파일 저장
    file_path = config.TRAINING_DIR / file.filename
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    return {
        "status": "success",
        "message": f"파일이 업로드되었습니다: {file.filename}",
        "file_path": str(file_path)
    }

@app.post("/upload/input")
async def upload_input_file(file: UploadFile = File(...)):
    """
    입력용 MIDI 파일 업로드
    
    업로드된 파일은 입력 디렉토리에 저장됩니다.
    """
    # 파일 확장자 확인
    if not file.filename.lower().endswith(('.mid', '.midi')):
        raise HTTPException(status_code=400, detail="MIDI 파일만 업로드 가능합니다.")
    
    # 파일 저장
    file_path = config.INPUT_DIR / file.filename
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    return {
        "status": "success",
        "message": f"파일이 업로드되었습니다: {file.filename}",
        "file_path": str(file_path)
    }

@app.post("/generate/{input_file}")
async def generate_midi(input_file: str, request: GenerateRequest, background_tasks: BackgroundTasks):
    """
    입력 MIDI 파일을 기반으로 새로운 MIDI 생성
    
    Args:
        input_file: 입력 MIDI 파일 이름 (입력 디렉토리 내)
        request: 생성 요청 정보
    """
    # 입력 파일 경로 확인
    input_path = config.INPUT_DIR / input_file
    if not os.path.exists(input_path):
        raise HTTPException(status_code=404, detail=f"입력 파일을 찾을 수 없습니다: {input_file}")
    
    # 출력 파일 경로 설정
    output_filename = f"generated_{os.path.splitext(input_file)[0]}.{request.output_format}"
    output_path = config.OUTPUT_DIR / output_filename
    
    # 백그라운드에서 MIDI 생성 실행
    async def generate_task():
        try:
            # MIDI 생성
            midi_data = await midi_rag_system.generate(
                str(input_path),
                output_format=request.output_format,
                genre_text=request.genre,
                lyrics_text=request.lyrics
            )
            
            # 생성된 MIDI 저장
            midi_rag_system.save_midi(midi_data, str(output_path))
        except Exception as e:
            print(f"MIDI 생성 중 오류 발생: {str(e)}")
    
    background_tasks.add_task(generate_task)
    
    return {
        "status": "generating",
        "message": "MIDI 생성이 시작되었습니다.",
        "input_file": str(input_path),
        "output_file": str(output_path)
    }

@app.get("/files/training")
async def list_training_files():
    """학습 디렉토리의 MIDI 파일 목록 조회"""
    midi_files = glob.glob(str(config.TRAINING_DIR / "*.mid")) + glob.glob(str(config.TRAINING_DIR / "*.midi"))
    return {"files": [os.path.basename(f) for f in midi_files]}

@app.get("/files/input")
async def list_input_files():
    """입력 디렉토리의 MIDI 파일 목록 조회"""
    midi_files = glob.glob(str(config.INPUT_DIR / "*.mid")) + glob.glob(str(config.INPUT_DIR / "*.midi"))
    return {"files": [os.path.basename(f) for f in midi_files]}

@app.get("/files/output")
async def list_output_files():
    """출력 디렉토리의 파일 목록 조회"""
    output_files = glob.glob(str(config.OUTPUT_DIR / "*.*"))
    return {"files": [os.path.basename(f) for f in output_files]}

@app.get("/download/{file_type}/{filename}")
async def download_file(file_type: str, filename: str):
    """
    파일 다운로드
    
    Args:
        file_type: 파일 유형 (training, input, output)
        filename: 파일 이름
    """
    # 파일 유형에 따른 디렉토리 설정
    if file_type == "training":
        file_dir = config.TRAINING_DIR
    elif file_type == "input":
        file_dir = config.INPUT_DIR
    elif file_type == "output":
        file_dir = config.OUTPUT_DIR
    else:
        raise HTTPException(status_code=400, detail=f"유효하지 않은 파일 유형: {file_type}")
    
    # 파일 경로 확인
    file_path = file_dir / filename
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"파일을 찾을 수 없습니다: {filename}")
    
    return FileResponse(path=file_path, filename=filename)

def start_server():
    """FastAPI 서버 시작"""
    uvicorn.run(
        "midi_rag_system.api.server:app",
        host=config.API_HOST,
        port=config.API_PORT,
        reload=config.API_DEBUG
    )

if __name__ == "__main__":
    start_server()
