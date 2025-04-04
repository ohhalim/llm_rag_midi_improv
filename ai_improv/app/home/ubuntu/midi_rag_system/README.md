# MIDI RAG 시스템 배포 패키지

이 패키지는 MCP, LangChain, FastAPI, YuE를 활용한 MIDI 기반 AI 즉흥 연주 시스템을 포함합니다.

## 패키지 내용

- `midi_rag_system/`: 시스템 소스 코드
  - `api/`: FastAPI 웹 서버
  - `core/`: 핵심 기능 모듈
  - `models/`: 모델 관련 모듈
  - `utils/`: 유틸리티 함수
  - `tests/`: 테스트 코드
- `data/`: 데이터 디렉토리
  - `input/`: 입력 MIDI 파일
  - `output/`: 출력 MIDI 파일
  - `training/`: 학습용 MIDI 파일
- `docs/`: 문서
  - `user_guide.md`: 사용자 가이드
  - `api_docs.md`: API 문서
  - `architecture.md`: 시스템 아키텍처 문서
- `main.py`: 명령줄 인터페이스
- `setup.py`: 설치 스크립트
- `config.py`: 시스템 설정
- `requirements.txt`: 의존성 목록

## 설치 방법

1. 저장소 클론 또는 패키지 다운로드:
```bash
git clone https://github.com/your-username/midi-rag-system.git
cd midi-rag-system
```

2. 설치 스크립트 실행:
```bash
python setup.py --env-file
```

CUDA 지원 및 FlashAttention 설치를 위한 추가 옵션:
```bash
python setup.py --cuda --flash-attn --env-file
```

## 시스템 실행

1. API 서버 실행:
```bash
python main.py server
```

2. MCP 서버 실행 (선택 사항):
```bash
python main.py mcp
```

3. 시스템 학습:
```bash
python main.py train
```

4. MIDI 생성:
```bash
python main.py generate <입력_파일> --output <출력_파일> --format midi --genre "장르: 재즈, 스타일: 스윙" --lyrics "즉흥 연주"
```

## 문서

자세한 내용은 다음 문서를 참조하세요:
- `docs/user_guide.md`: 시스템 사용 방법
- `docs/api_docs.md`: API 엔드포인트 설명
- `docs/architecture.md`: 시스템 아키텍처 설명

## 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.

YuE 모델은 Apache 2.0 라이선스 하에 배포됩니다.
