# MIDI RAG 시스템 사용자 가이드

## 개요

MIDI RAG(Retrieval-Augmented Generation) 시스템은 MCP, LangChain, FastAPI, YuE를 활용하여 MIDI 파일을 기반으로 한 AI 즉흥 연주 프로그램입니다. 이 시스템은 사용자가 입력한 MIDI 파일과 유사한 특성을 가진 MIDI 파일을 검색하고, 이를 기반으로 새로운 MIDI 음악을 생성합니다.

## 설치 방법

### 요구 사항

- Python 3.8 이상
- CUDA 11.8 이상 (GPU 사용 시)
- 최소 8GB RAM
- YuE 모델 실행을 위한 최소 24GB GPU 메모리 (권장)

### 설치 단계

1. 저장소 클론:
```bash
git clone https://github.com/your-username/midi-rag-system.git
cd midi-rag-system
```

2. 가상 환경 생성 및 활성화:
```bash
conda create -n midi-rag python=3.8
conda activate midi-rag
```

3. 의존성 설치:
```bash
pip install -r requirements.txt
```

4. (선택 사항) FlashAttention 2 설치 (GPU 메모리 사용량 감소):
```bash
pip install flash-attn --no-build-isolation
```

## 디렉토리 구조

```
midi_rag_system/
├── api/                # FastAPI 웹 서버
│   └── server.py       # API 엔드포인트 구현
├── core/               # 핵심 기능 모듈
│   ├── llm_api.py      # LLM API 모듈
│   ├── mcp_client.py   # MCP 클라이언트
│   ├── mcp_server.py   # MCP 서버
│   ├── midi_feature_extractor.py  # MIDI 특징 추출기
│   ├── midi_rag_system.py         # 메인 시스템 모듈
│   └── midi_vectorizer.py         # MIDI 벡터화 모듈
├── models/             # 모델 관련 모듈
│   └── yue_generator.py  # YuE 음악 생성 모듈
├── utils/              # 유틸리티 함수
├── data/               # 데이터 디렉토리
│   ├── input/          # 입력 MIDI 파일
│   ├── output/         # 출력 MIDI 파일
│   └── training/       # 학습용 MIDI 파일
├── tests/              # 테스트 코드
├── config.py           # 시스템 설정
└── requirements.txt    # 의존성 목록
```

## 환경 설정

시스템 설정은 `.env` 파일 또는 환경 변수를 통해 구성할 수 있습니다. 주요 설정 항목은 다음과 같습니다:

```
# API 설정
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
VECTORSTORE_PATH=/path/to/vectorstore
```

## 사용 방법

### 1. API 서버 실행

```bash
python -m midi_rag_system.api.server
```

서버가 시작되면 `http://localhost:8080`에서 API에 접근할 수 있습니다.

### 2. 학습용 MIDI 파일 업로드

학습용 MIDI 파일을 업로드하여 시스템이 참조할 수 있는 음악 데이터베이스를 구축합니다.

```bash
curl -X POST -F "file=@path/to/your/midi/file.mid" http://localhost:8080/upload/training
```

또는 웹 브라우저에서 `http://localhost:8080/docs`에 접속하여 Swagger UI를 통해 파일을 업로드할 수 있습니다.

### 3. 시스템 학습

업로드한 MIDI 파일을 사용하여 시스템을 학습시킵니다.

```bash
curl -X POST http://localhost:8080/train
```

### 4. 입력 MIDI 파일 업로드

생성의 기반이 될 입력 MIDI 파일을 업로드합니다.

```bash
curl -X POST -F "file=@path/to/your/input.mid" http://localhost:8080/upload/input
```

### 5. MIDI 생성

입력 MIDI 파일을 기반으로 새로운 MIDI 파일을 생성합니다.

```bash
curl -X POST -H "Content-Type: application/json" \
  -d '{"genre": "장르: 팝, 스타일: 현대적", "lyrics": "즉흥 연주를 위한 멜로디", "output_format": "midi"}' \
  http://localhost:8080/generate/input.mid
```

### 6. 생성된 MIDI 파일 다운로드

생성된 MIDI 파일을 다운로드합니다.

```bash
curl -X GET http://localhost:8080/download/output/generated_input.mid -o downloaded.mid
```

## API 엔드포인트

### 기본 정보

- `GET /`: API 상태 확인
- `GET /files/training`: 학습 디렉토리의 MIDI 파일 목록 조회
- `GET /files/input`: 입력 디렉토리의 MIDI 파일 목록 조회
- `GET /files/output`: 출력 디렉토리의 파일 목록 조회

### 파일 관리

- `POST /upload/training`: 학습용 MIDI 파일 업로드
- `POST /upload/input`: 입력용 MIDI 파일 업로드
- `GET /download/{file_type}/{filename}`: 파일 다운로드

### 시스템 기능

- `POST /train`: 학습용 MIDI 파일로 시스템 학습
- `POST /generate/{input_file}`: 입력 MIDI 파일을 기반으로 새로운 MIDI 생성

## 고급 사용법

### MCP 서버 활성화

MCP 서버를 활성화하여 외부 데이터 소스와 AI 모델 간의 연결을 개선할 수 있습니다.

```bash
# .env 파일에서 MCP 활성화
MCP_ENABLED=True

# MCP 서버 수동 실행 (필요한 경우)
python -c "from midi_rag_system.core.mcp_server import MIDIMCPServer; server = MIDIMCPServer(); server.start()"
```

### YuE 모델 커스터마이징

YuE 모델의 매개변수를 조정하여 생성되는 음악의 특성을 변경할 수 있습니다.

```bash
# .env 파일에서 YuE 설정 조정
YUE_MAX_NEW_TOKENS=4000  # 더 긴 음악 생성
YUE_REPETITION_PENALTY=1.2  # 반복 감소
```

## 문제 해결

### 일반적인 문제

1. **메모리 부족 오류**
   - YuE 모델은 많은 GPU 메모리를 필요로 합니다. 24GB 미만의 GPU를 사용하는 경우 `run_n_segments`와 `stage2_batch_size` 값을 줄여보세요.

2. **벡터 저장소 로드 실패**
   - 벡터 저장소가 없거나 손상된 경우 `/train` 엔드포인트를 호출하여 새로운 벡터 저장소를 생성하세요.

3. **MIDI 파일 처리 오류**
   - 일부 MIDI 파일은 형식이 잘못되었거나 지원되지 않는 기능을 사용할 수 있습니다. 표준 MIDI 파일을 사용해보세요.

### 로그 확인

문제 해결을 위해 로그를 확인할 수 있습니다.

```bash
# 로그 레벨 설정
export LOG_LEVEL=DEBUG

# 서버 실행 시 로그 출력
python -m midi_rag_system.api.server
```

## 성능 최적화

### GPU 메모리 사용량 최적화

- FlashAttention 2를 설치하여 GPU 메모리 사용량을 줄일 수 있습니다.
- 여러 GPU가 있는 경우 텐서 병렬 처리를 활성화할 수 있습니다.

### 처리 속도 향상

- `stage2_batch_size` 값을 증가시켜 처리 속도를 향상시킬 수 있습니다 (GPU 메모리가 충분한 경우).
- 벡터 저장소 크기를 최적화하여 검색 속도를 향상시킬 수 있습니다.

## 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 자세한 내용은 LICENSE 파일을 참조하세요.

YuE 모델은 Apache 2.0 라이선스 하에 배포됩니다.
