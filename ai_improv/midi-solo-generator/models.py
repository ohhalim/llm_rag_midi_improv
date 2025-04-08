from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, ARRAY, Text
from sqlalchemy.orm import relationship
from sqlalchemy.types import TypeDecorator
import json
from datetime import datetime

from database import Base

class JsonList(TypeDecorator):
    """SQLite는 ARRAY 타입을 지원하지 않으므로 대신 JSON 문자열 사용"""
    impl = Text
    
    def process_bind_param(self, value, dialect):
        if value is not None:
            return json.dumps(value)
        return None
    
    def process_result_value(self, value, dialect):
        if value is not None:
            return json.loads(value)
        return []

class MIDIFile(Base):
    __tablename__ = "midi_files"
    
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, index=True)
    path = Column(String)
    description = Column(String, nullable=True)
    tags = Column(JsonList)
    upload_date = Column(DateTime, default=datetime.now)
    source_file_id = Column(Integer, ForeignKey("midi_files.id"), nullable=True)
    
    derived_files = relationship("MIDIFile", backref="source_file", remote_side=[id])

class YueModel(Base):
    __tablename__ = "yue_models"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    path = Column(String)
    description = Column(String, nullable=True)
    upload_date = Column(DateTime, default=datetime.now)
    is_active = Column(Integer, default=0)  # 0: 비활성, 1: 활성

class ProcessingJob(Base):
    __tablename__ = "processing_jobs"
    
    id = Column(Integer, primary_key=True, index=True)
    input_file_id = Column(Integer, ForeignKey("midi_files.id"))
    output_file_id = Column(Integer, ForeignKey("midi_files.id"), nullable=True)
    job_type = Column(String)  # "improvisation", "fl_export" 등
    status = Column(String)  # "pending", "processing", "completed", "failed"
    created_at = Column(DateTime, default=datetime.now)
    completed_at = Column(DateTime, nullable=True)
    parameters = Column(JsonList)  # 추가 매개변수를 JSON으로 저장
    error_message = Column(String, nullable=True)
    
    input_file = relationship("MIDIFile", foreign_keys=[input_file_id])
    output_file = relationship("MIDIFile", foreign_keys=[output_file_id])