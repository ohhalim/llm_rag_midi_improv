"""
MIDI RAG 시스템 초기화 모듈
패키지 초기화 및 필요한 모듈 임포트를 설정합니다.
"""
# 버전 정보
__version__ = "1.0.0"

# 주요 모듈 임포트
from midi_rag_system.core.midi_rag_system import MIDIRAGSystem
from midi_rag_system.core.midi_feature_extractor import MIDIFeatureExtractor
from midi_rag_system.core.midi_vectorizer import MIDIVectorizer
from midi_rag_system.core.llm_api import LLMAPI
from midi_rag_system.models.yue_generator import YuEMusicGenerator

# API 서버 시작 함수 임포트
from midi_rag_system.api.server import start_server
