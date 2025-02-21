# midi_vectorizer.py
from typing import List, Dict
import os
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from midi_feature_extractor import MIDIFeatureExtractor

class MIDIVectorizer:
    def __init__(self):
        self.embeddings = OllamaEmbeddings(
            model="llama3.2",
            base_url="http://localhost:11434"
        )
        self.feature_extractor = MIDIFeatureExtractor()
    
    def _create_document(self, features: Dict, midi_file: str) -> Document:
        """특징을 Document 객체로 변환"""
        # 더 자세한 특징들을 포함하도록 수정
        feature_text = f"""
        파일명: {midi_file}
        
        기본 특징:
        - 템포: {features.get('tempo', [])}
        - 조표: {features.get('key_signatures', [])}
        - 박자: {features.get('time_signatures', [])}
        - 사용된 악기: {features.get('instruments', [])}
        
        멜로디 특징:
        - 음표 수: {len(features.get('notes', []))}
        - 음높이 범위: {features.get('pitch_range', [])}
        - 평균 음길이: {features.get('avg_note_duration', 0)}
        - 평균 세기: {features.get('avg_velocity', 0)}
        
        구조적 특징:
        - 트랙 수: {len(features.get('tracks', []))}
        - 총 길이: {features.get('total_duration', 0)} 초
        - 코드 진행: {features.get('chord_progression', [])}
        """
        
        return Document(
            page_content=feature_text,
            metadata={
                "filename": midi_file,
                "features": features,
                "musical_style": self._analyze_style(features)  # 음악 스타일 분석 추가
            }
        )
    
    def _analyze_style(self, features: Dict) -> str:
        """MIDI 특징을 기반으로 음악 스타일 분석"""
        # 여기에 스타일 분석 로직 추가
        # 예: 템포, 악기, 코드 진행 등을 기반으로 장르나 스타일 추정
        return "분석된 스타일"
    
    def vectorize_midi(self, midi_files: List[str]):
        """MIDI 파일들을 벡터화하여 저장"""
        docs = []
        for midi_file in midi_files:
            try:
                features = self.feature_extractor.extract_features(midi_file)
                doc = self._create_document(features, midi_file)
                docs.append(doc)
            except Exception as e:
                print(f"Error processing {midi_file}: {str(e)}")
        
        vectorstore = FAISS.from_documents(documents=docs, embedding=self.embeddings)
        return vectorstore
    
    def save_vectorstore(self, vectorstore, save_path: str):
        """벡터 저장소를 파일로 저장"""
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            vectorstore.save_local(save_path)
            print(f"벡터 저장소가 {save_path}에 저장되었습니다.")
            return True
        except Exception as e:
            print(f"벡터 저장소 저장 중 오류 발생: {str(e)}")
            return False
    
    def load_vectorstore(self, load_path: str):
        """저장된 벡터 저장소 로드"""
        try:
            if os.path.exists(load_path):
                vectorstore = FAISS.load_local(load_path, self.embeddings)
                print(f"벡터 저장소를 {load_path}에서 로드했습니다.")
                return vectorstore
            else:
                print(f"벡터 저장소 파일을 찾을 수 없습니다: {load_path}")
                return None
        except Exception as e:
            print(f"벡터 저장소 로드 중 오류 발생: {str(e)}")
            return None

# Enumerating objects: 28, done.
# Counting objects: 100% (28/28), done.
# Delta compression using up to 8 threads
# Compressing objects: 100% (16/16), done.
# Writing objects: 100% (18/18), 6.02 KiB | 6.02 MiB/s, done.
# Total 18 (delta 6), reused 14 (delta 2), pack-reused 0
# remote: Resolving deltas: 100% (6/6), completed with 5 local objects.
# To https://github.com/ohhalim/llm_rag_midi_improv.git
#    b140a99..687a08a  main -> main
# (venv) (base) ohhalim@MacBookAir llm_rag_midi_improv %  python /Users/ohhalim/git_box/llm_rag_midi_improv/ai_improv/test.py    

# === 1. 특징 추출 테스트 ===
# {
#   "file": "ai_improv/data/training/\ub9c8\uc74c \uc194\ub85c\ub77c\uc778.mid",
#   "features": {
#     "tempo": {
#       "main_tempo": 71.0,
#       "tempo_changes": 1
#     },
#     "harmony": {
#       "chord_progression": [
#         "Perfect Twelfth",
#         "Diminished Sixth",
#         "Perfect 26th",
#         "Perfect Eleventh",
#         "enharmonic equivalent to major triad",
#         "Major 23rd"
#       ],
#       "unique_chords": 6
#     },
#     "rhythm": {
#       "avg_note_duration": 1.2168615984405458,
#       "rhythmic_density": 171
#     },
#     "melody": {
#       "pitch_range": [
#         33,
#         85
#       ],
#       "avg_pitch": 57.559322033898304
#     },
#     "key_signatures": [],
#     "time_signatures": [
#       "4/4"
#     ],
#     "instruments": [
#       "None"
#     ]
#   }
# }

# === 2. 벡터 저장소 생성 테스트 (처음 5개 파일) ===
# {
#   "files": [
#     "ai_improv/data/training/144114.mid",
#     "ai_improv/data/training/\u110b\u116f\u11bf \u1106\u1175\u1103\u1175.mid",
#     "ai_improv/data/training/5\u110b\u116f\u11af\u1103\u1161\u11af\u110b\u1166 \u110b\u1169\u1106\u1161\u110c\u116e\u1112\u1161\u1103\u1161\u1100\u1161 \u1106\u1161\u11ab\u1100\u1165.mid",
#     "ai_improv/data/training/\u1112\u1161\u1110\u1173\u1107\u1165\u11ab.mid",
#     "ai_improv/data/training/\u1112\u1166\u110b\u1175\u110c\u1173 \u110b\u1161\u110b\u1175\u11b7\u1111\u1161\u110b\u1175\u11ab.mid"
#   ]
# }

# === 벡터 저장소 생성 완료 ===
# 성공

# === 3. 유사도 검색 결과 ===
# [
#   {
#     "filename": "ai_improv/data/training/144114.mid",
#     "score": "N/A"
#   },
#   {
#     "filename": "ai_improv/data/training/\u110b\u116f\u11bf \u1106\u1175\u1103\u1175.mid",
#     "score": "N/A"
#   },
#   {
#     "filename": "ai_improv/data/training/\u1112\u1161\u1110\u1173\u1107\u1165\u11ab.mid",
#     "score": "N/A"
#   }
# ]

# === 4. LLM 생성 결과 ===
# ```json
# {
#     "tracks": [
#         {
#             "instrument": 0,
#             "notes": [
#                 {"pitch": 57, "time": 0, "duration": 1.2, "velocity": 64},
#                 {"pitch": 60, "time": 1.2, "duration": 1.2, "velocity": 32},
#                 {"pitch": 63, "time": 2.4, "duration": 1.2, "velocity": 127},
#                 {"pitch": 66, "time": 3.6, "duration": 1.2, "velocity": 90},
#                 {"pitch": 69, "time": 5.0, "duration": 1.2, "velocity": 64},
#                 {"pitch": 72, "time": 6.4, "duration": 1.2, "velocity": 32},
#                 {"pitch": 75, "time": 7.8, "duration": 1.2, "velocity": 127},
#                 {"pitch": 78, "time": 9.0, "duration": 1.2, "velocity": 90},
#                 {"pitch": 81, "time": 10.4, "duration": 1.2, "velocity": 64},
#                 {"pitch": 84, "time": 11.8, "duration": 1.2, "velocity": 32}
#             ]
#         }
#     ],
#     "time_signatures": ["4/4"],
#     "key_signatures": []
# }
# ```

# 위 JSON은 최소 100개 이상의 음표가 포함된 MIDI를 생성합니다. 음높이는 33에서 85 사이이며, 평균 음표 길이는 1.2초와 비슷합니다. 시작 시간은 이전 음표의 (시작 시간 + 지속 시간)으로 계산됩니다. velocity는 32에서 127 사이의 다양한 값입니다.



# 전처리 , 청크 ,파싱 , 