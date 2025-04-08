"""
MIDI RAG 시스템의 메인 모듈 - 솔로라인 생성 특화 버전
MCP, LangChain, YuE를 통합하여 MIDI 기반 AI 솔로라인 생성 시스템을 구현합니다.
"""
from typing import List, Dict, Any, Optional
import os
import json
from pathlib import Path
import glob
import logging

import config
from midi_rag_system.core.midi_feature_extractor import MIDIFeatureExtractor
from midi_rag_system.core.midi_vectorizer import MIDIVectorizer
from midi_rag_system.core.llm_api import LLMAPI
from midi_rag_system.core.mcp_client import MIDIMCPClient
from midi_rag_system.models.yue_generator import YuESoloGenerator

# 로깅 설정
logging.basicConfig(level=getattr(logging, config.LOG_LEVEL, "INFO"), 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("MIDIRAGSystem")

class MIDIRAGSystem:
    def __init__(self):
        """MIDI RAG 시스템 초기화 - 솔로라인 생성 특화"""
        self.vectorizer = MIDIVectorizer()
        self.llm_api = LLMAPI()
        self.feature_extractor = MIDIFeatureExtractor()
        self.yue_generator = YuESoloGenerator()
        self.vectorstore = None
        
        # MCP 클라이언트 초기화
        self.mcp_client = MIDIMCPClient()
        
        logger.info("MIDI RAG 시스템 초기화 완료 (솔로라인 생성 특화)")
    
    def train(self, midi_files: List[str], save_path: str = None):
        """
        MIDI 파일들로 RAG 시스템 학습 및 벡터 저장소 저장
        
        Args:
            midi_files: 학습에 사용할 MIDI 파일 경로 목록
            save_path: 벡터 저장소를 저장할 경로 (선택적)
        """
        logger.info(f"벡터 저장소 생성 중... (파일 {len(midi_files)}개)")
        self.vectorstore = self.vectorizer.vectorize_midi(midi_files)
        
        # 벡터 저장소 저장
        if save_path:
            self.vectorizer.save_vectorstore(self.vectorstore, save_path)
            logger.info(f"벡터 저장소가 {save_path}에 저장되었습니다.")
    
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
    
    async def generate(self, 
                 input_midi: str, 
                 reference_midi: str = None,
                 genre_text: str = None, 
                 lyrics_text: str = None,
                 solo_instrument: int = 0):
        """
        입력 MIDI에 어울리는 솔로라인 생성
        
        Args:
            input_midi (str): 입력 MIDI 파일 경로
            reference_midi (str): 참조 MIDI 파일 경로 (선택적)
            genre_text (str): 장르 및 스타일 텍스트 (선택적)
            lyrics_text (str): 가사 텍스트 (선택적)
            solo_instrument (int): 솔로 악기 번호 (기본값: 0, 피아노)
            
        Returns:
            bytes: 생성된 MIDI 파일 바이트
        """
        try:
            # 벡터 저장소가 없는 경우 경고
            if not self.vectorstore:
                logger.warning("벡터 저장소가 없습니다. 벡터 검색 없이 진행합니다.")
            
            # 입력 MIDI 특징 추출
            logger.info(f"입력 MIDI 분석 중: {input_midi}")
            input_features = self.feature_extractor.extract_features(input_midi)
            
            # 유사한 MIDI 참조 검색 (벡터 저장소가 있는 경우)
            similar_features = []
            if self.vectorstore:
                # 로컬 벡터 저장소에서 유사한 MIDI 찾기
                logger.info("유사한 MIDI 패턴 검색 중...")
                similar_docs = self.vectorstore.similarity_search(str(input_features), k=3)
                similar_features = [doc.page_content for doc in similar_docs]
                
                # 유사한 MIDI 파일 이름 출력
                logger.info("유사한 MIDI 파일:")
                for i, doc in enumerate(similar_docs):
                    logger.info(f"  {i+1}. {doc.metadata.get('filename', 'Unknown')}")
            
            # 참조 MIDI 분석 (제공된 경우)
            if reference_midi and os.path.exists(reference_midi):
                logger.info(f"참조 MIDI 분석 중: {reference_midi}")
                reference_features = self.feature_extractor.extract_features(reference_midi)
                # 참조 특징을 유사 특징 목록에 추가 (더 높은 가중치 부여)
                if reference_features:
                    similar_features.insert(0, str(reference_features))
            
            # 기본 장르 텍스트 설정
            if not genre_text:
                if solo_instrument == 0:  # 피아노
                    genre_text = "장르: 재즈, 스타일: 피아노 솔로, 악기: 피아노"
                elif solo_instrument in [24, 25, 26]:  # 기타
                    genre_text = "장르: 록, 스타일: 기타 솔로, 악기: 일렉 기타"
                elif solo_instrument in [65, 66, 67]:  # 색소폰 
                    genre_text = "장르: 재즈, 스타일: 색소폰 솔로, 악기: 색소폰"
                else:
                    genre_text = "장르: 팝, 스타일: 솔로 라인, 악기: 리드 악기"
            
            # 기본 가사 텍스트 설정 (YuE 모델이 텍스트 형식의 프롬프트를 받음)
            if not lyrics_text:
                lyrics_text = """[verse]
즉흥 솔로 연주 부분

[chorus]
악기의 표현력 있는 솔로"""
            
            # 두 가지 방식으로 솔로라인 생성 시도
            try:
                # 1. YuE를 사용한 솔로라인 생성
                logger.info("YuE 모델을 사용하여 솔로라인 생성 중...")
                midi_data = self.yue_generator.generate_solo(
                    input_midi=input_midi,
                    reference_midi=reference_midi,
                    genre_text=genre_text,
                    lyrics_text=lyrics_text,
                    solo_instrument=solo_instrument
                )
                
                # MIDI 데이터가 생성되었는지 확인
                if not midi_data:
                    raise Exception("YuE에서 유효한 MIDI 데이터를 생성하지 못했습니다.")
                
                logger.info("YuE 모델을 통한 솔로라인 생성 완료")
                return midi_data
                
            except Exception as yue_error:
                # YuE 생성이 실패한 경우 대체 방법 시도
                logger.warning(f"YuE 생성 실패: {str(yue_error)}, LLM API 사용 시도")
                
                # 2. LLM API를 사용한 솔로라인 생성
                logger.info("LLM API를 사용하여 솔로라인 생성 중...")
                
                # LLM API에 입력 및 유사 특징 전달
                json_response = self.llm_api.generate_response(
                    str(input_features),
                    "\n\n".join(similar_features[:3])  # 최대 3개의 유사 특징 사용
                )
                
                # 솔로라인 확장
                enhanced_json = self.llm_api.enhance_solo(json_response, min_notes=100)
                
                # JSON을 MIDI로 변환
                midi_data = self.json_to_midi(enhanced_json)
                
                logger.info("LLM API를 통한 솔로라인 생성 완료")
                return midi_data
            
        except Exception as e:
            logger.error(f"솔로라인 생성 중 오류 발생: {str(e)}")
            # 오류 발생 시 기본 솔로라인 생성
            return self.generate_fallback_solo(solo_instrument)
    
    def json_to_midi(self, json_data):
        """
        JSON 형식의 MIDI 데이터를 MIDI 바이트로 변환
        
        Args:
            json_data: JSON 문자열 또는 딕셔너리
            
        Returns:
            bytes: MIDI 파일 바이트
        """
        # JSON 문자열인 경우 파싱
        if isinstance(json_data, str):
            try:
                data = json.loads(json_data)
            except json.JSONDecodeError:
                logger.error("유효하지 않은 JSON 데이터")
                return self.generate_fallback_solo()
        else:
            data = json_data
        
        return self.yue_generator.from_json(json.dumps(data))
    
    def generate_fallback_solo(self, instrument=0):
        """
        기본 솔로라인 생성 (다른 방법이 실패한 경우)
        
        Args:
            instrument: 솔로 악기 번호 (기본값: 0, 피아노)
            
        Returns:
            bytes: MIDI 파일 바이트
        """
        logger.info(f"기본 솔로라인 생성 (악기: {instrument})")
        return self.yue_generator._generate_fallback_solo(instrument)
    
    def save_midi(self, midi_data, output_path):
        """
        생성된 MIDI 데이터를 파일로 저장
        
        Args:
            midi_data: MIDI 바이트 데이터
            output_path: 저장할 파일 경로
        """
        try:
            # 출력 디렉토리 확인
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # MIDI 바이너리 데이터 저장
            with open(output_path, 'wb') as f:
                f.write(midi_data)
            
            logger.info(f"MIDI 파일이 저장되었습니다: {output_path}")
            return True
        except Exception as e:
            logger.error(f"MIDI 파일 저장 중 오류 발생: {str(e)}")
            return False