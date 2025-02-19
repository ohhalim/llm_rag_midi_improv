from midi_feature_extractor import MIDIFeatureExtractor
from midi_vectorizer import MIDIVectorizer
from llm_api import LLMAPI

class MIDIRAGSystem:
    def __init__(self):
        self.vectorizer = MIDIVectorizer()
        self.llm_api = LLMAPI()
        
    def train(self, midi_files: List[str]):
        """MIDI 파일들로 RAG 시스템 학습"""
        self.vectorstore = self.vectorizer.vectorize_midi(midi_files)
    
    def generate(self, input_midi: str) -> str:
        """입력 MIDI에 어울리는 새로운 MIDI 생성"""
        # 입력 MIDI 특징 추출
        input_features = self.vectorizer.feature_extractor.extract_features(input_midi)
        
        # 유사한 MIDI 찾기
        similar_docs = self.vectorstore.similarity_search(str(input_features))
        
        # 새로운 MIDI 생성
        response = self.llm_api.generate_response(input_features, similar_docs)
        
        return response