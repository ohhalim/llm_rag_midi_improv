"""
MIDI RAG 시스템 설치 스크립트
시스템 설치 및 초기 설정을 자동화합니다.
"""
import os
import sys
import subprocess
import argparse
from pathlib import Path

def parse_args():
    """명령줄 인수 파싱"""
    parser = argparse.ArgumentParser(description="MIDI RAG 시스템 설치")
    parser.add_argument("--cuda", action="store_true", help="CUDA 지원 설치")
    parser.add_argument("--flash-attn", action="store_true", help="FlashAttention 설치")
    parser.add_argument("--env-file", action="store_true", help=".env 파일 생성")
    parser.add_argument("--m1-optimize", action="store_true", help="M1 MacBook 최적화")
    return parser.parse_args()

def create_env_file():
    """기본 .env 파일 생성"""
    env_content = """# API 설정
API_HOST=0.0.0.0
API_PORT=8080
API_DEBUG=False

# MCP 설정
MCP_ENABLED=True
MCP_SERVER_HOST=localhost
MCP_SERVER_PORT=8000

# YuE 모델 설정
YUE_STAGE1_MODEL=m-a-p/YuE-s1-7B-anneal-en-cot
YUE_STAGE2_MODEL=m-a-p/YuE-s2-1B-general
YUE_MAX_NEW_TOKENS=3000
YUE_REPETITION_PENALTY=1.1

# Ollama 설정
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2

# 벡터 저장소 설정
VECTORSTORE_PATH=./data/vectorstore
"""
    
    with open(".env", "w") as f:
        f.write(env_content)
    
    print("기본 .env 파일이 생성되었습니다.")

def apply_m1_optimization():
    """M1 MacBook 최적화 적용"""
    print("M1 MacBook 최적화 적용 중...")
    
    # .env 파일에 M1 최적화 설정 추가
    env_content = """
# M1 최적화 설정
M1_OPTIMIZE=True
YUE_LOAD_IN_8BIT=True
YUE_RUN_N_SEGMENTS=1
YUE_STAGE2_BATCH_SIZE=1
YUE_USE_FLASH_ATTENTION=False
"""
    
    with open(".env", "a") as f:
        f.write(env_content)
    
    # 필요한 패키지 설치
    subprocess.run([
        sys.executable, "-m", "pip", "install", 
        "bitsandbytes", "accelerate", "psutil"
    ])
    
    print("M1 MacBook 최적화가 적용되었습니다.")

def install_requirements(cuda=False, flash_attn=False):
    """요구 사항 설치"""
    print("의존성 설치 중...")
    
    # 기본 요구 사항 설치
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    
    # CUDA 지원 설치
    if cuda:
        print("CUDA 지원 설치 중...")
        subprocess.run([
            sys.executable, "-m", "pip", "install", 
            "torch", "torchvision", "torchaudio", 
            "--index-url", "https://download.pytorch.org/whl/cu118"
        ])
    
    # FlashAttention 설치
    if flash_attn:
        print("FlashAttention 설치 중...")
        subprocess.run([
            sys.executable, "-m", "pip", "install", 
            "flash-attn", "--no-build-isolation"
        ])
    
    print("의존성 설치가 완료되었습니다.")

def create_directories():
    """필요한 디렉토리 생성"""
    print("디렉토리 구조 생성 중...")
    
    # 데이터 디렉토리 생성
    os.makedirs("data/input", exist_ok=True)
    os.makedirs("data/output", exist_ok=True)
    os.makedirs("data/training", exist_ok=True)
    os.makedirs("data/vectorstore", exist_ok=True)
    
    print("디렉토리 구조가 생성되었습니다.")

def main():
    """메인 함수"""
    args = parse_args()
    
    # 현재 디렉토리 확인
    if not os.path.exists("requirements.txt"):
        print("오류: requirements.txt 파일을 찾을 수 없습니다.")
        print("MIDI RAG 시스템 루트 디렉토리에서 이 스크립트를 실행하세요.")
        return
    
    # 디렉토리 생성
    create_directories()
    
    # 요구 사항 설치
    install_requirements(cuda=args.cuda, flash_attn=args.flash_attn)
    
    # .env 파일 생성
    if args.env_file:
        create_env_file()
    
    # M1 최적화 적용
    if args.m1_optimize:
        apply_m1_optimization()
    
    print("\nMIDI RAG 시스템 설치가 완료되었습니다!")
    print("\n시스템 실행 방법:")
    print("1. API 서버 실행: python main.py server")
    print("2. MCP 서버 실행: python main.py mcp")
    print("3. 시스템 학습: python main.py train")
    print("4. MIDI 생성: python main.py generate <입력_파일>")
    print("\n자세한 내용은 docs/user_guide.md 파일을 참조하세요.")

if __name__ == "__main__":
    main()