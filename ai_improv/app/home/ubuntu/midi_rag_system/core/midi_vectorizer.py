"""
MIDI 벡터화 모듈
MIDI 특징을 벡터로 변환하고 벡터 저장소를 관리합니다.
"""
from typing import List, Dict
import os
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from midi_rag_system.core.midi_feature_extractor import MIDIFeatureExtractor
import config

class MIDIVectorizer:
    def __init__(self, base_url=None, model=None):
        """
        MIDI 벡터화 모듈 초기화
        
        Args:
            base_url: Ollama 서버 URL (기본값: config.py에서 설정)
            model: 임베딩에 사용할 모델 (기본값: config.py에서 설정)
        """
        self.embeddings = OllamaEmbeddings(
            model=model or config.OLLAMA_MODEL,
            base_url=base_url or config.OLLAMA_BASE_URL
        )
        self.feature_extractor = MIDIFeatureExtractor()
    
    def _create_document(self, features: Dict, midi_file: str) -> Document:
        """
        특징을 Document 객체로 변환
        
        Args:
            features: MIDI 특징 딕셔너리
            midi_file: MIDI 파일 경로
            
        Returns:
            Document: LangChain Document 객체
        """
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
        """
        MIDI 특징을 기반으로 음악 스타일 분석
        
        Args:
            features: MIDI 특징 딕셔너리
            
        Returns:
            str: 분석된 음악 스타일
        """
        # 여기에 스타일 분석 로직 추가
        # 예: 템포, 악기, 코드 진행 등을 기반으로 장르나 스타일 추정
        return "분석된 스타일"
    
    def vectorize_midi(self, midi_files: List[str]):
        """
        MIDI 파일들을 벡터화하여 저장
        
        Args:
            midi_files: MIDI 파일 경로 목록
            
        Returns:
            FAISS: 벡터 저장소 객체
        """
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
    
    def save_vectorstore(self, vectorstore, save_path: str = None):
        """
        벡터 저장소를 파일로 저장
        
        Args:
            vectorstore: 저장할 벡터 저장소 객체
            save_path: 저장 경로 (기본값: config.py에서 설정)
            
        Returns:
            bool: 저장 성공 여부
        """
        save_path = save_path or config.VECTORSTORE_PATH
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            vectorstore.save_local(save_path)
            print(f"벡터 저장소가 {save_path}에 저장되었습니다.")
            return True
        except Exception as e:
            print(f"벡터 저장소 저장 중 오류 발생: {str(e)}")
            return False
    
    def load_vectorstore(self, load_path: str = None):
        """
        저장된 벡터 저장소 로드
        
        Args:
            load_path: 로드할 벡터 저장소 경로 (기본값: config.py에서 설정)
            
        Returns:
            FAISS: 로드된 벡터 저장소 객체 또는 None
        """
        load_path = load_path or config.VECTORSTORE_PATH
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
