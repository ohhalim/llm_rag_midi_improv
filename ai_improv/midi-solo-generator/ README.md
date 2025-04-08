# MIDI 즉흥연주 솔로 생성 시스템

이 프로젝트는 YuE 기반 음악 데이터를 사용하여 입력 MIDI 파일에 맞는 즉흥 솔로 라인을 생성하는 시스템입니다.

## 기능
- MIDI 파일 업로드 및 관리
- 다양한 스타일(재즈, 펑크, 비밥, 블루스)의 솔로 라인 생성
- FL Studio 연동을 통한 오디오 출력
- YuE 모델 관리

## 설치
1. 의존성 설치: `pip install -r requirements.txt`
2. YuE 기본 모델 다운로드 후 `models/` 디렉토리에 저장

## 실행
```bash
uvicorn app:app --reload