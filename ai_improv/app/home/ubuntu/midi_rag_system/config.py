"""
MIDI RAG 시스템의 메인 설정 파일 - 솔로라인 생성 특화 버전
환경 변수 및 기본 설정을 관리합니다.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# 기본 경로 설정
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
INPUT_DIR = DATA_DIR / "input"
OUTPUT_DIR = DATA_DIR / "output"
TRAINING_DIR = DATA_DIR / "training"

# 디렉토리가 없으면 생성
for dir_path in [INPUT_DIR, OUTPUT_DIR, TRAINING_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# API 키 및 환경 설정
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")

# MCP 설정
MCP_ENABLED = os.getenv("MCP_ENABLED", "True").lower() in ("true", "1", "t")
MCP_SERVER_HOST = os.getenv("MCP_SERVER_HOST", "localhost")
MCP_SERVER_PORT = int(os.getenv("MCP_SERVER_PORT", "8000"))

# YuE 모델 설정
YUE_STAGE1_MODEL = os.getenv("YUE_STAGE1_MODEL", "m-a-p/YuE-s1-7B-anneal-en-cot")
YUE_STAGE2_MODEL = os.getenv("YUE_STAGE2_MODEL", "m-a-p/YuE-s2-1B-general")
YUE_MAX_NEW_TOKENS = int(os.getenv("YUE_MAX_NEW_TOKENS", "3000"))
YUE_REPETITION_PENALTY = float(os.getenv("YUE_REPETITION_PENALTY", "1.1"))
YUE_DIR = os.getenv("YUE_DIR", os.path.expanduser("~/YuE/inference"))

# FastAPI 설정
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8080"))
API_DEBUG = os.getenv("API_DEBUG", "False").lower() in ("true", "1", "t")

# FL Studio 연결 설정
FL_STUDIO_PLUGIN_PORT = int(os.getenv("FL_STUDIO_PLUGIN_PORT", "9000"))
FL_STUDIO_CALLBACK_URL = os.getenv("FL_STUDIO_CALLBACK_URL", f"http://localhost:{FL_STUDIO_PLUGIN_PORT}/callback")

# 벡터 저장소 설정
VECTORSTORE_PATH = os.getenv("VECTORSTORE_PATH", str(DATA_DIR / "vectorstore"))

# 로깅 설정
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = os.getenv("LOG_FILE", str(BASE_DIR / "logs" / "midi_rag.log"))

# 최적화 설정
YUE_LOAD_IN_8BIT = os.getenv("YUE_LOAD_IN_8BIT", "True").lower() in ("true", "1", "t")
YUE_LOAD_IN_4BIT = os.getenv("YUE_LOAD_IN_4BIT", "False").lower() in ("true", "1", "t")
YUE_RUN_N_SEGMENTS = int(os.getenv("YUE_RUN_N_SEGMENTS", "1"))
YUE_STAGE2_BATCH_SIZE = int(os.getenv("YUE_STAGE2_BATCH_SIZE", "1"))
YUE_USE_FLASH_ATTENTION = os.getenv("YUE_USE_FLASH_ATTENTION", "False").lower() in ("true", "1", "t")

# 시스템 임계값 설정
MAX_INPUT_FILE_SIZE_MB = int(os.getenv("MAX_INPUT_FILE_SIZE_MB", "10"))  # 최대 입력 파일 크기 (MB)
MAX_GENERATION_TIME_SEC = int(os.getenv("MAX_GENERATION_TIME_SEC", "600"))  # 최대 생성 시간 (초)
MIN_NOTES_IN_SOLO = int(os.getenv("MIN_NOTES_IN_SOLO", "100"))  # 솔로에 필요한 최소 노트 수