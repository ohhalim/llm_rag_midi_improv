"""
FastAPI 웹 서버 모듈 - FL Studio 통합 버전
MIDI RAG 시스템의 API 엔드포인트를 구현하고 FL Studio와 연결합니다.
"""
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import glob
import asyncio
from typing import List, Optional, Dict, Any
import shutil
from pydantic import BaseModel
import json
import time

import config
from midi_rag_system.core.midi_rag_system import MIDIRAGSystem
from midi_rag_system.core.mcp_client import MIDIMCPClient

# FastAPI 앱 초기화
app = FastAPI(
    title="MIDI 솔로라인 생성 API",
    description="YuE, LangChain, MCP를 활용한 MIDI 기반 AI 솔로라인 생성 시스템",
    version="1.0.0"
)

# CORS 설정 (FL Studio 웹 인터페이스와의 통신 허용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MIDI RAG 시스템 초기화
midi_rag_system = MIDIRAGSystem()

# MCP 클라이언트 초기화
mcp_client = MIDIMCPClient()

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
    solo_instrument: int = 0
    use_reference: bool = False
    reference_file: Optional[str] = None

# FL Studio 연결을 위한 상태 추적
class GenerationStatus:
    def __init__(self):
        self.active_generations = {}  # task_id -> status
        self.completed_outputs = {}   # task_id -> output_file_path

# 상태 객체 생성
generation_status = GenerationStatus()

@app.get("/")
async def root():
    """API 상태 확인"""
    return {"status": "online", "message": "MIDI 솔로라인 생성 API가 실행 중입니다."}

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
    입력 MIDI 파일을 기반으로 솔로라인 생성
    
    Args:
        input_file: 입력 MIDI 파일 이름 (입력 디렉토리 내)
        request: 생성 요청 정보
    """
    # 입력 파일 경로 확인
    input_path = config.INPUT_DIR / input_file
    if not os.path.exists(input_path):
        raise HTTPException(status_code=404, detail=f"입력 파일을 찾을 수 없습니다: {input_file}")
    
    # 참조 파일 경로 확인 (사용하는 경우)
    reference_path = None
    if request.use_reference and request.reference_file:
        reference_path = config.INPUT_DIR / request.reference_file
        if not os.path.exists(reference_path):
            raise HTTPException(status_code=404, detail=f"참조 파일을 찾을 수 없습니다: {request.reference_file}")
    
    # 출력 파일 경로 설정
    timestamp = int(time.time())
    output_filename = f"solo_{os.path.splitext(input_file)[0]}_{timestamp}.mid"
    output_path = config.OUTPUT_DIR / output_filename
    
    # 태스크 ID 생성
    task_id = f"task_{timestamp}"
    
    # 상태 초기화
    generation_status.active_generations[task_id] = {
        "status": "generating",
        "progress": 0,
        "input_file": str(input_path),
        "output_file": str(output_path),
        "start_time": time.time()
    }
    
    # 백그라운드에서 솔로라인 생성 실행
    async def generate_task():
        try:
            # 상태 업데이트
            generation_status.active_generations[task_id]["progress"] = 10
            
            # MCP 클라이언트를 통해 FL Studio로 작업 상태 전송
            await mcp_client.send_status_update({
                "task_id": task_id,
                "status": "processing",
                "progress": 10,
                "message": "입력 MIDI 분석 중..."
            })
            
            # MIDI 생성
            generation_status.active_generations[task_id]["progress"] = 30
            await mcp_client.send_status_update({
                "task_id": task_id,
                "status": "processing",
                "progress": 30,
                "message": "솔로라인 생성 중..."
            })
            
            midi_data = await midi_rag_system.generate(
                str(input_path),
                reference_midi=str(reference_path) if reference_path else None,
                genre_text=request.genre,
                lyrics_text=request.lyrics,
                solo_instrument=request.solo_instrument
            )
            
            # 생성된 MIDI 저장
            generation_status.active_generations[task_id]["progress"] = 80
            await mcp_client.send_status_update({
                "task_id": task_id,
                "status": "processing",
                "progress": 80,
                "message": "생성된 솔로라인 저장 중..."
            })
            
            midi_rag_system.save_midi(midi_data, str(output_path))
            
            # 상태 업데이트
            generation_status.active_generations[task_id]["status"] = "completed"
            generation_status.active_generations[task_id]["progress"] = 100
            generation_status.active_generations[task_id]["completion_time"] = time.time()
            
            # 완료된 출력 저장
            generation_status.completed_outputs[task_id] = str(output_path)
            
            # MCP를 통해 FL Studio로 완료 알림
            await mcp_client.send_output_ready_notification({
                "task_id": task_id,
                "status": "completed",
                "output_file": str(output_path),
                "message": "솔로라인 생성 완료"
            })
            
        except Exception as e:
            # 오류 상태 업데이트
            generation_status.active_generations[task_id]["status"] = "failed"
            generation_status.active_generations[task_id]["error"] = str(e)
            
            # MCP를 통해 FL Studio로 오류 알림
            await mcp_client.send_status_update({
                "task_id": task_id,
                "status": "failed",
                "error": str(e),
                "message": "솔로라인 생성 중 오류 발생"
            })
            
            print(f"솔로라인 생성 중 오류 발생: {str(e)}")
    
    background_tasks.add_task(generate_task)
    
    return {
        "task_id": task_id,
        "status": "generating",
        "message": "솔로라인 생성이 시작되었습니다.",
        "input_file": str(input_path),
        "output_file": str(output_path)
    }

@app.get("/status/{task_id}")
async def get_generation_status(task_id: str):
    """생성 작업의 현재 상태를 확인합니다."""
    if task_id not in generation_status.active_generations:
        raise HTTPException(status_code=404, detail=f"작업 ID를 찾을 수 없습니다: {task_id}")
    
    status = generation_status.active_generations[task_id]
    
    # 작업이 완료된 경우 출력 파일 정보 추가
    if status["status"] == "completed" and task_id in generation_status.completed_outputs:
        status["output_file"] = generation_status.completed_outputs[task_id]
    
    return status

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
    
    return FileResponse(path=file_path, filename=filename, media_type="audio/midi")

@app.get("/latest_solo")
async def get_latest_solo():
    """가장 최근에 생성된 솔로라인 파일을 반환합니다. FL Studio와의 빠른 통합을 위한 헬퍼."""
    output_files = glob.glob(str(config.OUTPUT_DIR / "*.mid")) + glob.glob(str(config.OUTPUT_DIR / "*.midi"))
    
    if not output_files:
        raise HTTPException(status_code=404, detail="생성된 솔로라인 파일이 없습니다.")
    
    # 가장 최근 파일 찾기 (수정 시간 기준)
    latest_file = max(output_files, key=os.path.getmtime)
    
    return FileResponse(
        path=latest_file, 
        filename=os.path.basename(latest_file),
        media_type="audio/midi"
    )

# MCP 특화 엔드포인트 - FL Studio 연결용
@app.post("/mcp/register_client")
async def register_fl_studio_client(client_info: Dict[str, Any]):
    """FL Studio 클라이언트 등록 (MCP 연결)"""
    try:
        # MCP 클라이언트에 등록
        result = await mcp_client.register_client(client_info)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"FL Studio 클라이언트 등록 중 오류 발생: {str(e)}")

@app.post("/mcp/import_to_fl_studio/{task_id}")
async def import_to_fl_studio(task_id: str):
    """생성된 솔로라인을 FL Studio로 직접 가져오기"""
    if task_id not in generation_status.completed_outputs:
        raise HTTPException(status_code=404, detail=f"완료된 작업을 찾을 수 없습니다: {task_id}")
    
    output_path = generation_status.completed_outputs[task_id]
    
    try:
        # MCP를 통해 FL Studio로 MIDI 전송
        result = await mcp_client.send_midi_to_fl_studio({
            "file_path": output_path,
            "track_name": f"AI Solo {os.path.basename(output_path)}",
            "create_new_track": True
        })
        
        return {
            "status": "success",
            "message": "솔로라인이 FL Studio로 가져와졌습니다.",
            "details": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"FL Studio로 가져오기 중 오류 발생: {str(e)}")

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