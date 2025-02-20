# midi_vectorizer.py
from typing import List, Dict
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