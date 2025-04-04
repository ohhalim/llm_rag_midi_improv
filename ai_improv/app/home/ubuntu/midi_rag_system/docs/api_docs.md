# MIDI RAG 시스템 API 문서

이 문서는 MIDI RAG 시스템의 API 엔드포인트에 대한 상세 설명을 제공합니다.

## 기본 정보

- **기본 URL**: `http://localhost:8080`
- **응답 형식**: JSON
- **인증**: 필요 없음

## 엔드포인트

### 시스템 상태 확인

```
GET /
```

시스템의 현재 상태를 확인합니다.

#### 응답

```json
{
  "status": "online",
  "message": "MIDI RAG API가 실행 중입니다."
}
```

### 학습 디렉토리의 MIDI 파일 목록 조회

```
GET /files/training
```

학습 디렉토리에 있는 모든 MIDI 파일의 목록을 반환합니다.

#### 응답

```json
{
  "files": ["example1.mid", "example2.mid", "example3.mid"]
}
```

### 입력 디렉토리의 MIDI 파일 목록 조회

```
GET /files/input
```

입력 디렉토리에 있는 모든 MIDI 파일의 목록을 반환합니다.

#### 응답

```json
{
  "files": ["input1.mid", "input2.mid"]
}
```

### 출력 디렉토리의 파일 목록 조회

```
GET /files/output
```

출력 디렉토리에 있는 모든 파일의 목록을 반환합니다.

#### 응답

```json
{
  "files": ["generated_input1.mid", "generated_input1.json"]
}
```

### 학습용 MIDI 파일 업로드

```
POST /upload/training
```

학습용 MIDI 파일을 업로드합니다.

#### 요청

- **Content-Type**: `multipart/form-data`
- **Body**:
  - `file`: MIDI 파일 (`.mid` 또는 `.midi` 확장자)

#### 응답

```json
{
  "status": "success",
  "message": "파일이 업로드되었습니다: example.mid",
  "file_path": "/home/user/midi_rag_system/data/training/example.mid"
}
```

### 입력용 MIDI 파일 업로드

```
POST /upload/input
```

입력용 MIDI 파일을 업로드합니다.

#### 요청

- **Content-Type**: `multipart/form-data`
- **Body**:
  - `file`: MIDI 파일 (`.mid` 또는 `.midi` 확장자)

#### 응답

```json
{
  "status": "success",
  "message": "파일이 업로드되었습니다: input.mid",
  "file_path": "/home/user/midi_rag_system/data/input/input.mid"
}
```

### 파일 다운로드

```
GET /download/{file_type}/{filename}
```

지정된 유형의 디렉토리에서 파일을 다운로드합니다.

#### 매개변수

- **file_type**: 파일 유형 (`training`, `input`, `output` 중 하나)
- **filename**: 다운로드할 파일 이름

#### 응답

요청한 파일의 바이너리 데이터가 반환됩니다.

### 시스템 학습

```
POST /train
```

학습 디렉토리의 MIDI 파일을 사용하여 시스템을 학습시킵니다.

#### 응답

```json
{
  "status": "training",
  "message": "학습이 시작되었습니다. 5개의 MIDI 파일을 처리합니다.",
  "files": [
    "/home/user/midi_rag_system/data/training/example1.mid",
    "/home/user/midi_rag_system/data/training/example2.mid",
    "..."
  ]
}
```

### MIDI 생성

```
POST /generate/{input_file}
```

입력 MIDI 파일을 기반으로 새로운 MIDI를 생성합니다.

#### 매개변수

- **input_file**: 입력 디렉토리에 있는 MIDI 파일 이름

#### 요청

- **Content-Type**: `application/json`
- **Body**:
  ```json
  {
    "genre": "장르: 팝, 스타일: 현대적",
    "lyrics": "즉흥 연주를 위한 멜로디",
    "output_format": "midi"
  }
  ```
  - **genre**: (선택 사항) 생성할 음악의 장르 및 스타일 설명
  - **lyrics**: (선택 사항) 생성할 음악의 가사 또는 설명
  - **output_format**: 출력 형식 (`midi` 또는 `json`)

#### 응답

```json
{
  "status": "generating",
  "message": "MIDI 생성이 시작되었습니다.",
  "input_file": "/home/user/midi_rag_system/data/input/input.mid",
  "output_file": "/home/user/midi_rag_system/data/output/generated_input.mid"
}
```

## 오류 응답

API는 다음과 같은 형식으로 오류를 반환합니다:

```json
{
  "detail": "오류 메시지"
}
```

### 일반적인 오류 코드

- **400 Bad Request**: 잘못된 요청 형식 또는 매개변수
- **404 Not Found**: 요청한 리소스를 찾을 수 없음
- **500 Internal Server Error**: 서버 내부 오류

## 예제

### 학습 예제

```bash
# 학습용 MIDI 파일 업로드
curl -X POST -F "file=@example.mid" http://localhost:8080/upload/training

# 시스템 학습
curl -X POST http://localhost:8080/train
```

### 생성 예제

```bash
# 입력 MIDI 파일 업로드
curl -X POST -F "file=@input.mid" http://localhost:8080/upload/input

# MIDI 생성
curl -X POST -H "Content-Type: application/json" \
  -d '{"genre": "장르: 재즈, 스타일: 스윙", "lyrics": "즉흥 연주", "output_format": "midi"}' \
  http://localhost:8080/generate/input.mid

# 생성된 MIDI 파일 다운로드
curl -X GET http://localhost:8080/download/output/generated_input.mid -o result.mid
```
