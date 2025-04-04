"""
MIDI RAG 시스템의 메인 모듈
MCP, LangChain, YuE를 통합하여 MIDI 기반 AI 즉흥 연주 시스템을 구현합니다.
"""
from typing import List, Dict, Any, Optional
import os
import json
from pathlib import Path
import glob

import config
from midi_rag_system.core.midi_feature_extractor import MIDIFeatureExtractor
from midi_rag_system.core.midi_vectorizer import MIDIVectorizer
from midi_rag_system.core.llm_api import LLMAPI
from midi_rag_system.core.mcp_client import MIDIMCPClient
from midi_rag_system.models.yue_generator import YuEMusicGenerator

class MIDIRAGSystem:
    def __init__(self):
        """MIDI RAG 시스템 초기화"""
        self.vectorizer = MIDIVectorizer()
        self.llm_api = LLMAPI()
        self.feature_extractor = MIDIFeatureExtractor()
        self.yue_generator = YuEMusicGenerator()
        self.vectorstore = None
        
        # MCP 클라이언트 초기화 (MCP가 활성화된 경우)
        self.mcp_client = None
        if config.MCP_ENABLED:
            self.mcp_client = MIDIMCPClient()
    
    def train(self, midi_files: List[str], save_path: str = None):
        """
        MIDI 파일들로 RAG 시스템 학습 및 벡터 저장소 저장
        
        Args:
            midi_files: 학습에 사용할 MIDI 파일 경로 목록
            save_path: 벡터 저장소를 저장할 경로 (선택적)
        """
        print(f"벡터 저장소 생성 중... (파일 {len(midi_files)}개)")
        self.vectorstore = self.vectorizer.vectorize_midi(midi_files)
        
        # 벡터 저장소 저장
        if save_path:
            self.vectorizer.save_vectorstore(self.vectorstore, save_path)
            print(f"벡터 저장소가 {save_path}에 저장되었습니다.")
    
    def load_vectorstore(self, load_path: str):
        """
        저장된 벡터 저장소 로드
        
        Args:
            load_path: 벡터 저장소가 저장된 경로
        
        Returns:
            bool: 로드 성공 여부
        """
        self.vectorstore = self.vectorizer.load_vectorstore(load_path)
        return self.vectorstore is not None
    
    async def generate(self, input_midi: str, output_format='midi', genre_text=None, lyrics_text=None):
        """
        입력 MIDI에 어울리는 새로운 MIDI 생성
        
        Args:
            input_midi (str): 입력 MIDI 파일 경로
            output_format (str): 출력 형식 ('json' 또는 'midi')
            genre_text (str): 장르 및 스타일 텍스트 (선택적)
            lyrics_text (str): 가사 텍스트 (선택적)
            
        Returns:
            str 또는 bytes: 'json' 형식이면 JSON 문자열, 'midi' 형식이면 MIDI 파일 바이트
        """
        # 벡터 저장소가 없는 경우 오류
        if not self.vectorstore and not config.MCP_ENABLED:
            raise ValueError("벡터 저장소가 없습니다. train() 메소드를 호출하거나 load_vectorstore()로 저장소를 로드하세요.")
        
        # 입력 MIDI 특징 추출
        input_features = self.feature_extractor.extract_features(input_midi)
        
        # MCP가 활성화된 경우 MCP 서버를 통해 유사한 MIDI 검색
        if config.MCP_ENABLED and self.mcp_client:
            search_result = await self.mcp_client.search_midi(str(input_features), k=3)
            similar_features = [doc.get("content", "") for doc in search_result.get("results", [])]
            
            # 유사한 MIDI 파일 이름 출력
            print("유사한 MIDI 파일:")
            for i, doc in enumerate(search_result.get("results", [])):
                print(f"  {i+1}. {doc.get('filename', 'Unknown')}")
        else:
            # 로컬 벡터 저장소에서 유사한 MIDI 찾기
            similar_docs = self.vectorstore.similarity_search(str(input_features), k=3)
            similar_features = [doc.page_content for doc in similar_docs]
            
            # 유사한 MIDI 파일 이름 출력
            print("유사한 MIDI 파일:")
            for i, doc in enumerate(similar_docs):
                print(f"  {i+1}. {doc.metadata.get('filename', 'Unknown')}")
        
        # LLM API를 통한 MIDI 생성 (JSON 형식)
        if not genre_text:
            genre_text = "장르: 팝, 스타일: 현대적, 악기: 피아노, 기타, 드럼"
        
        if not lyrics_text:
            lyrics_text = "즉흥 연주를 위한 멜로디"
        
        # YuE 모델을 사용하여 음악 생성
        if output_format == 'json':
            # JSON 형식은 문자열이 아닌 바이트로 변환하여 반환
            if isinstance(json_output, str):
                return json_output.encode('utf-8')
            return json_output
        else:
            # MIDI 형식은 이미 바이트 객체
            return midi_output
        
    def save_midi(self, midi_data, output_path):
        """
        생성된 MIDI 데이터를 파일로 저장
        
        Args:
            midi_data: MIDI 데이터 (JSON 문자열 또는 바이너리 데이터)
            output_path: 저장할 파일 경로
        """
        # 출력 디렉토리 확인
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        if isinstance(midi_data, str):
            # JSON 형식인 경우
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(midi_data)
        else:
            # MIDI 바이너리 데이터인 경우
            with open(output_path, 'wb') as f:
                f.write(midi_data)
        
        print(f"MIDI 파일이 저장되었습니다: {output_path}")
