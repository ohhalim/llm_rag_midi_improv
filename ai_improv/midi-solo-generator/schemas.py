from pydantic import BaseModel
from typing import List, Optional, Any
from datetime import datetime

class MIDIFileBase(BaseModel):
    filename: str
    description: Optional[str] = None
    tags: List[str] = []

class MIDIFileCreate(MIDIFileBase):
    pass

class MIDIFile(MIDIFileBase):
    id: int
    path: str
    upload_date: datetime
    source_file_id: Optional[int] = None
    
    class Config:
        orm_mode = True

class YueModelBase(BaseModel):
    name: str
    description: Optional[str] = None

class YueModelCreate(YueModelBase):
    pass

class YueModel(YueModelBase):
    id: int
    path: str
    upload_date: datetime
    is_active: int
    
    class Config:
        orm_mode = True

class ProcessingJobBase(BaseModel):
    input_file_id: int
    job_type: str
    parameters: Optional[Any] = None

class ProcessingJobCreate(ProcessingJobBase):
    pass

class ProcessingJob(ProcessingJobBase):
    id: int
    output_file_id: Optional[int] = None
    status: str
    created_at: datetime
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    
    class Config:
        orm_mode = True