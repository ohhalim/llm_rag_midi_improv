import os
import json
from midi_rag import MIDIRAGSystem
from midi_feature_extractor import MIDIFeatureExtractor
from midi_vectorizer import MIDIVectorizer
from llm_api import LLMAPI

# 디버그 로깅 함수
def log_step(step_name, data, save_to_file=False):
    print(f"\n=== {step_name} ===")
    formatted_data = json.dumps(data, indent=2, ensure_ascii=False) if isinstance(data, (dict, list)) else data
    print(formatted_data)
    
    if save_to_file:
        with open(f"test_results_{step_name.replace(' ', '_')}.json", "w", encoding="utf-8") as f:
            f.write(formatted_data)

def main():
    # 경로 설정
    base_dir = os.path.dirname(os.path.abspath(__file__))
    training_dir = os.path.join(base_dir, "data", "training")
    output_dir = os.path.join(base_dir, "data", "output")
    
    # 출력 디렉토리 확인
    os.makedirs(output_dir, exist_ok=True)
    
    # 테스트용 MIDI 파일 선택
    try:
        midi_files = [f for f in os.listdir(training_dir) if f.endswith('.mid')]
        if not midi_files:
            print(f"Error: No MIDI files found in {training_dir}")
            return
        
        test_file = os.path.join(training_dir, midi_files[0])
        log_step("테스트 파일", test_file)
    except Exception as e:
        print(f"Error accessing training directory: {str(e)}")
        return
    
    # 1. 특징 추출기 초기화 및 테스트
    try:
        feature_extractor = MIDIFeatureExtractor()
        features = feature_extractor.extract_features(test_file)
        log_step("1. 특징 추출 테스트", {
            "file": test_file,
            "features": features
        }, save_to_file=True)
    except Exception as e:
        log_step("특징 추출 실패", str(e))
        return
    
    # 2. 벡터 저장소 생성 테스트
    try:
        vectorizer = MIDIVectorizer()
        training_files = [os.path.join(training_dir, f) for f in midi_files[:5]]  # 처음 5개 파일만 테스트
        
        log_step("2. 벡터 저장소 생성 테스트", {
            "files": training_files
        })
        
        vectorstore = vectorizer.vectorize_midi(training_files)
        log_step("벡터 저장소 생성 완료", "성공")
    except Exception as e:
        log_step("벡터 저장소 생성 실패", str(e))
        return
    
    # 3. 유사도 검색 테스트
    try:
        query_features = feature_extractor.extract_features(test_file)
        similar_docs = vectorstore.similarity_search(str(query_features), k=3)
        
        similarity_results = [
            {
                "filename": doc.metadata["filename"],
                "score": doc.metadata.get("score", "N/A"),
                "musical_style": doc.metadata.get("musical_style", "N/A"),
                "snippet": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
            }
            for doc in similar_docs
        ]
        
        log_step("3. 유사도 검색 결과", similarity_results, save_to_file=True)
    except Exception as e:
        log_step("유사도 검색 실패", str(e))
        return
    
    # 4. LLM 초기화 및 테스트
    try:
        llm_api = LLMAPI()
        
        # 입력 특징과 유사한 특징을 문자열로 변환
        input_features_str = json.dumps(query_features, ensure_ascii=False, indent=2)
        similar_features_str = "\n".join([doc.page_content for doc in similar_docs])
        
        # 응답 생성
        response = llm_api.generate_response(input_features_str, similar_features_str)
        log_step("4. LLM 생성 결과", response, save_to_file=True)
        
        # JSON 유효성 검사
        try:
            generated_json = json.loads(response)
            track_count = len(generated_json.get("tracks", []))
            note_count = sum(len(track.get("notes", [])) for track in generated_json.get("tracks", []))
            
            log_step("생성된 MIDI 통계", {
                "track_count": track_count,
                "note_count": note_count,
                "time_signatures": generated_json.get("time_signatures", []),
                "key_signatures": generated_json.get("key_signatures", [])
            })
        except json.JSONDecodeError:
            log_step("JSON 파싱 실패", "생성된 응답이 유효한 JSON이 아닙니다")
    except Exception as e:
        log_step("LLM 생성 실패", str(e))
        return
    
    # 5. 전체 RAG 시스템 테스트
    try:
        log_step("5. 전체 RAG 시스템 테스트", "시작")
        
        rag_system = MIDIRAGSystem()
        rag_system.train(training_files)
        
        output_midi = rag_system.generate(test_file)
        output_path = os.path.join(output_dir, "generated_output.json")
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(output_midi)
        
        log_step("전체 RAG 시스템 테스트 완료", {
            "output_saved_to": output_path
        })
    except Exception as e:
        log_step("전체 RAG 시스템 테스트 실패", str(e))

if __name__ == "__main__":
    main()