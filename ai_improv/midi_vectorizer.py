from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

class MIDIVectorizer:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings()
        self.feature_extractor = MIDIFeatureExtractor()
    
    def vectorize_midi(self, midi_files: List[str]):
        """MIDI 파일들을 벡터화하여 저장"""
        docs = []
        for midi_file in midi_files:
            features = self.feature_extractor.extract_features(midi_file)
            doc = self._create_document(features, midi_file)
            docs.append(doc)
        
        vectorstore = FAISS.from_documents(documents=docs, embedding=self.embeddings)
        return vectorstore
    
    def _create_document(self, features: Dict, midi_file: str) -> Document:
        # 특징을 Document 객체로 변환
        pass