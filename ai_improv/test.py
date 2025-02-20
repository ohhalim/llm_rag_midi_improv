import os
from midi_rag import MIDIRAGSystem
import json

# 디버그 로깅 함수
def log_step(step_name, data):
    print(f"\n=== {step_name} ===")
    print(json.dumps(data, indent=2) if isinstance(data, (dict, list)) else data)

# RAG 시스템 초기화
rag_system = MIDIRAGSystem()

# 테스트용 MIDI 파일 선택
training_dir = "ai_improv/data/training"
test_file = os.path.join(training_dir, "마음 솔로라인.mid")  # 예시 파일

# 1. 특징 추출 테스트
log_step("1. 특징 추출 테스트", {
    "file": test_file,
    "features": rag_system.feature_extractor.extract_features(test_file)
})

# 2. 벡터 저장소 생성 테스트
training_files = [os.path.join(training_dir, f) for f in os.listdir(training_dir) if f.endswith('.mid')][:5]  # 처음 5개 파일만 테스트
log_step("2. 벡터 저장소 생성 테스트 (처음 5개 파일)", {
    "files": training_files
})

try:
    rag_system.train(training_files)
    log_step("벡터 저장소 생성 완료", "성공")
except Exception as e:
    log_step("벡터 저장소 생성 실패", str(e))

# 3. 유사도 검색 테스트
try:
    input_features = rag_system.feature_extractor.extract_features(test_file)
    similar_docs = rag_system.vectorstore.similarity_search(str(input_features), k=3)
    log_step("3. 유사도 검색 결과", [
        {"filename": doc.metadata["filename"], "score": doc.metadata.get("score", "N/A")}
        for doc in similar_docs
    ])
except Exception as e:
    log_step("유사도 검색 실패", str(e))

# 4. LLM 생성 테스트
try:
    output_midi = rag_system.generate(test_file)
    log_step("4. LLM 생성 결과", output_midi)
except Exception as e:
    log_step("LLM 생성 실패", str(e))