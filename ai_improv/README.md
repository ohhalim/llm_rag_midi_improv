# MIDI 프로세서

MIDI 파일 입출력 및 변환을 위한 간단한 도구입니다.

## 기능

- MIDI 파일 로드 및 저장
- MIDI 파일을 JSON 형식으로 변환
- JSON 형식의 데이터를 MIDI 파일로 변환
- MIDI 파일 정보 조회

## 설치

```bash
# 저장소 클론
git clone https://github.com/username/midi-processor.git
cd midi-processor

# 필요한 패키지 설치
pip install -r requirements.txt
```

## 사용 방법

### 명령줄에서 사용

```bash
# MIDI 파일 정보 보기
python main.py --input sample.mid --info

# MIDI 파일을 JSON으로 변환
python main.py --input sample.mid --json sample.json

# MIDI 파일 복사
python main.py --input sample.mid --output output.mid

# MIDI 파일 로드, JSON 변환, 새 MIDI 파일 저장
python main.py --input sample.mid --json sample.json --output output.mid
```

### 코드에서 사용

```python
from midi_processor.core.processor import MIDIProcessor

# 프로세서 초기화
processor = MIDIProcessor()

# MIDI 파일 로드
processor.load("sample.mid")

# 정보 출력
info = processor.get_info()
print(info)

# JSON으로 변환
json_data = processor.to_json()
with open("sample.json", "w") as f:
    f.write(json_data)

# 다른 이름으로 저장
processor.save("output.mid")

# JSON에서 MIDI 생성
with open("sample.json", "r") as f:
    json_data = f.read()
processor.from_json(json_data)
processor.save("from_json.mid")
```

## 요구사항

- Python 3.7 이상
- mido
- numpy
- music21 