#!/bin/bash

# Docker 명령어 스크립트

# 현재 디렉토리
CURRENT_DIR=$(pwd)
echo "현재 디렉토리: $CURRENT_DIR"

# 명령어 파싱
if [ "$1" == "build" ]; then
    echo "Docker 이미지 빌드 중..."
    docker-compose build
    echo "빌드 완료!"

elif [ "$1" == "up" ]; then
    echo "Docker 컨테이너 시작 중..."
    docker-compose up -d
    echo "컨테이너가 백그라운드에서 실행 중입니다."
    echo "API 서버: http://localhost:8080"
    echo "MCP 서버: http://localhost:8000"

elif [ "$1" == "down" ]; then
    echo "Docker 컨테이너 중지 중..."
    docker-compose down
    echo "컨테이너가 중지되었습니다."

elif [ "$1" == "logs" ]; then
    if [ "$2" == "api" ]; then
        echo "API 서버 로그 출력 중..."
        docker logs -f midi_rag_api
    elif [ "$2" == "mcp" ]; then
        echo "MCP 서버 로그 출력 중..."
        docker logs -f midi_rag_mcp
    else
        echo "전체 로그 출력 중..."
        docker-compose logs -f
    fi

elif [ "$1" == "train" ]; then
    echo "MIDI 학습 실행 중..."
    docker run --rm -v $CURRENT_DIR/data:/app/data midi_rag_system train
    echo "학습 완료!"

elif [ "$1" == "generate" ]; then
    if [ -z "$2" ]; then
        echo "사용법: $0 generate <입력 MIDI 파일>"
        exit 1
    fi
    
    INPUT_FILE="$2"
    INPUT_BASENAME=$(basename "$INPUT_FILE")
    
    # 입력 파일 data/input 디렉토리로 복사
    mkdir -p data/input
    cp "$INPUT_FILE" "data/input/$INPUT_BASENAME"
    
    echo "MIDI 생성 실행 중..."
    docker run --rm -v $CURRENT_DIR/data:/app/data midi_rag_system generate "data/input/$INPUT_BASENAME"
    echo "생성 완료! 결과는 data/output 디렉토리에서 확인하세요."

elif [ "$1" == "shell" ]; then
    echo "Docker 컨테이너 쉘 실행 중..."
    if [ "$2" == "api" ]; then
        docker exec -it midi_rag_api /bin/bash
    elif [ "$2" == "mcp" ]; then
        docker exec -it midi_rag_mcp /bin/bash
    else
        docker run --rm -it midi_rag_system /bin/bash
    fi

else
    echo "MIDI RAG 시스템 Docker 명령어"
    echo "사용법: $0 [명령어] [옵션]"
    echo ""
    echo "명령어:"
    echo "  build       - Docker 이미지 빌드"
    echo "  up          - Docker 컨테이너 시작"
    echo "  down        - Docker 컨테이너 중지"
    echo "  logs [api|mcp] - 로그 출력"
    echo "  train       - MIDI 파일로 시스템 학습"
    echo "  generate <입력 MIDI 파일> - MIDI 생성"
    echo "  shell [api|mcp] - 컨테이너 쉘 실행"
fi 