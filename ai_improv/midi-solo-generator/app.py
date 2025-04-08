from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Query
from fastapi.responses import FileResponse, JSONResponse
from sqlalchemy.orm import Session
from typing import List, Optional
import os
import shutil
from datetime import datetime

from database import SessionLocal, engine
import models, schemas
from midi_processor import YueProcessor, MIDIGenerator, FLStudioInterface

# 모델 초기화
models.Base.metadata.create_all(bind=engine)

app = FastAPI(title="MIDI 즉흥연주 API")
improvisation_system = None

# 의존성
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# 시스템 초기화
@app.on_event("startup")
async def startup_event():
    global improvisation_system
    improvisation_system = {
        "yue": YueProcessor("model_files/yue_base_model"),
        "generator": MIDIGenerator("midi_database/"),
        "fl_interface": FLStudioInterface()
    }
    # 디렉토리 생성
    os.makedirs("uploads", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

# MIDI 파일 관리 엔드포인트
@app.post("/midi/upload/", response_model=schemas.MIDIFile)
async def upload_midi(
    file: UploadFile = File(...),
    tags: Optional[str] = Query(None, description="콤마로 구분된 태그"),
    description: Optional[str] = Query(None),
    db: Session = Depends(get_db)
):
    """MIDI 파일을 시스템에 업로드"""
    # 파일 저장
    file_path = f"uploads/{file.filename}"
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # DB 항목 생성
    tag_list = tags.split(",") if tags else []
    db_midi = models.MIDIFile(
        filename=file.filename,
        path=file_path,
        description=description,
        tags=tag_list,
        upload_date=datetime.now()
    )
    db.add(db_midi)
    db.commit()
    db.refresh(db_midi)
    
    return db_midi

@app.get("/midi/files/", response_model=List[schemas.MIDIFile])
def list_midi_files(
    skip: int = 0,
    limit: int = 100,
    tag: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """사용 가능한 모든 MIDI 파일 나열(필터링 가능)"""
    query = db.query(models.MIDIFile)
    if tag:
        query = query.filter(models.MIDIFile.tags.contains([tag]))
    return query.offset(skip).limit(limit).all()

@app.get("/midi/file/{file_id}", response_model=schemas.MIDIFile)
def get_midi_file(file_id: int, db: Session = Depends(get_db)):
    """특정 MIDI 파일에 대한 정보 가져오기"""
    db_midi = db.query(models.MIDIFile).filter(models.MIDIFile.id == file_id).first()
    if not db_midi:
        raise HTTPException(status_code=404, detail="MIDI 파일을 찾을 수 없음")
    return db_midi

@app.get("/midi/download/{file_id}")
def download_midi_file(file_id: int, db: Session = Depends(get_db)):
    """MIDI 파일 다운로드"""
    db_midi = db.query(models.MIDIFile).filter(models.MIDIFile.id == file_id).first()
    if not db_midi:
        raise HTTPException(status_code=404, detail="MIDI 파일을 찾을 수 없음")
    return FileResponse(db_midi.path, filename=db_midi.filename)

@app.delete("/midi/file/{file_id}")
def delete_midi_file(file_id: int, db: Session = Depends(get_db)):
    """MIDI 파일 삭제"""
    db_midi = db.query(models.MIDIFile).filter(models.MIDIFile.id == file_id).first()
    if not db_midi:
        raise HTTPException(status_code=404, detail="MIDI 파일을 찾을 수 없음")
    
    # 물리적 파일 삭제
    if os.path.exists(db_midi.path):
        os.remove(db_midi.path)
    
    # 데이터베이스 항목 삭제
    db.delete(db_midi)
    db.commit()
    
    return {"message": "MIDI 파일이 성공적으로 삭제됨"}

# YuE 기반 음악 데이터 엔드포인트
@app.post("/yue/upload-model/")
async def upload_yue_model(
    file: UploadFile = File(...),
    description: Optional[str] = Query(None),
    db: Session = Depends(get_db)
):
    """새 YuE 모델 파일 업로드"""
    # 파일 저장
    file_path = f"models/{file.filename}"
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # DB 항목 생성
    db_model = models.YueModel(
        name=file.filename.split(".")[0],
        path=file_path,
        description=description,
        upload_date=datetime.now()
    )
    db.add(db_model)
    db.commit()
    db.refresh(db_model)
    
    # 현재 모델 업데이트
    global improvisation_system
    improvisation_system["yue"] = YueProcessor(file_path)
    
    return {"message": "YuE 모델이 업로드되고 활성화됨", "model_id": db_model.id}

@app.get("/yue/models/", response_model=List[schemas.YueModel])
def list_yue_models(db: Session = Depends(get_db)):
    """사용 가능한 모든 YuE 모델 나열"""
    return db.query(models.YueModel).all()

@app.post("/yue/activate/{model_id}")
def activate_yue_model(model_id: int, db: Session = Depends(get_db)):
    """특정 YuE 모델 활성화"""
    db_model = db.query(models.YueModel).filter(models.YueModel.id == model_id).first()
    if not db_model:
        raise HTTPException(status_code=404, detail="YuE 모델을 찾을 수 없음")
    
    # 현재 모델 업데이트
    global improvisation_system
    improvisation_system["yue"] = YueProcessor(db_model.path)
    
    return {"message": f"YuE 모델 {db_model.name}이(가) 성공적으로 활성화됨"}

# 즉흥연주 엔드포인트
@app.post("/improvise/solo/{file_id}")
async def improvise_solo(
    file_id: int, 
    style: str = "jazz_funk",
    complexity: float = 0.7,
    length: int = 32,
    db: Session = Depends(get_db)
):
    """업로드된 MIDI 파일에 기반한 즉흥 솔로 라인 생성"""
    db_midi = db.query(models.MIDIFile).filter(models.MIDIFile.id == file_id).first()
    if not db_midi:
        raise HTTPException(status_code=404, detail="MIDI 파일을 찾을 수 없음")
    
    try:
        # YuE를 통한 처리
        with open(db_midi.path, "rb") as f:
            midi_data = f.read()
        
        yue_context = improvisation_system["yue"].process_input(midi_data)
        
        # 즉흥 솔로 생성
        generated_midi = improvisation_system["generator"].generate_solo(
            yue_context, 
            style=style,
            complexity=complexity,
            length=length
        )
        
        # 생성된 출력 저장
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"improv_{timestamp}.mid"
        output_path = f"outputs/{output_filename}"
        
        with open(output_path, "wb") as f:
            f.write(generated_midi)
        
        # 출력을 위한 DB 항목 생성
        db_output = models.MIDIFile(
            filename=output_filename,
            path=output_path,
            description=f"{style} 스타일의 {db_midi.filename} 기반 즉흥연주",
            tags=[style, "improvisation", "solo_line", "generated"],
            upload_date=datetime.now(),
            source_file_id=db_midi.id
        )
        db.add(db_output)
        db.commit()
        db.refresh(db_output)
        
        return {
            "message": "즉흥연주가 성공적으로 생성됨",
            "output_id": db_output.id,
            "download_url": f"/midi/download/{db_output.id}"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"즉흥연주 생성 실패: {str(e)}")

@app.post("/improvise/to-fl-studio/{file_id}")
async def send_to_fl_studio(
    file_id: int,
    db: Session = Depends(get_db)
):
    """MIDI 파일을 MCP를 통해 FL Studio로 전송"""
    db_midi = db.query(models.MIDIFile).filter(models.MIDIFile.id == file_id).first()
    if not db_midi:
        raise HTTPException(status_code=404, detail="MIDI 파일을 찾을 수 없음")
    
    try:
        # MIDI 파일 읽기
        with open(db_midi.path, "rb") as f:
            midi_data = f.read()
        
        # FL Studio로 전송
        improvisation_system["fl_interface"].send_midi(midi_data)
        
        return {"message": "MIDI가 FL Studio로 성공적으로 전송됨"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"FL Studio 통신 실패: {str(e)}")

# 시스템 상태 및 구성
@app.get("/system/status")
async def system_status():
    """시스템 구성 요소의 현재 상태 확인"""
    return {
        "yue_model": improvisation_system["yue"].get_model_info(),
        "midi_generator": improvisation_system["generator"].get_status(),
        "fl_studio": improvisation_system["fl_interface"].get_connection_status()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
