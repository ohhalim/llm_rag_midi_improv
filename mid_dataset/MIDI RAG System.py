from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
import mido
import numpy as np
from typing import List, Dict
import json

class MIDIFeatureExtractor:
    """MIDI 파일에서 특징을 추출하는 클래스"""
    def __init__(self):
        self.features = {}
    
    def extract_features(self, midi_file: str) -> Dict:
        """MIDI 파일에서 주요 특징들을 추출"""
        mid = mido.MidiFile(midi_file)
        
        # 기본 특징 추출
        features = {
            'tempo': [],
            'time_signature': [],
            'key_signature': [],
            'note_density': [],
            'pitch_range': [],
            'velocity_range': [],
            'duration_patterns': []
        }
        
        for track in mid.tracks:
            notes = []
            current_time = 0
            
            for msg in track:
                current_time += msg.time
                
                if msg.type == 'set_tempo':
                    features['tempo'].append(mido.tempo2bpm(msg.tempo))
                elif msg.type == 'time_signature':
                    features['time_signature'].append(f"{msg.numerator}/{msg.denominator}")
                elif msg.type == 'key_signature':
                    features['key_signature'].append(msg.key)
                elif msg.type == 'note_on' and msg.velocity > 0:
                    notes.append({
                        'pitch': msg.note,
                        'velocity': msg.velocity,
                        'time': current_time
                    })
            
            if notes:
                # 노트 밀도 계산
                total_time = current_time
                features['note_density'].append(len(notes) / total_time if total_time > 0 else 0)
                
                # 피치 범위
                pitches = [note['pitch'] for note in notes]
                features['pitch_range'].append({
                    'min': min(pitches),
                    'max': max(pitches)
                })
                
                # 벨로시티 범위
                velocities = [note['velocity'] for note in notes]
                features['velocity_range'].append({
                    'min': min(velocities),
                    'max': max(velocities)
                })
        
        return features

class MIDIVectorizer:
    """MIDI 특징을 벡터로 변환하는 클래스"""
    def __init__(self, embeddings_model=None):
        self.embeddings_model = embeddings_model or OpenAIEmbeddings()
        self.vector_store = None
    
    def vectorize_features(self, features: Dict) -> str:
        """특징을 텍스트로 변환"""
        feature_text = f"""
        Tempo: {np.mean(features['tempo']) if features['tempo'] else 'Unknown'}
        Time Signatures: {', '.join(set(features['time_signature']))}
        Key Signatures: {', '.join(set(features['key_signature']))}
        Average Note Density: {np.mean(features['note_density'])}
        Pitch Range: {self._format_range(features['pitch_range'])}
        Velocity Range: {self._format_range(features['velocity_range'])}
        """
        return feature_text.strip()
    
    def _format_range(self, ranges: List[Dict]) -> str:
        if not ranges:
            return "Unknown"
        min_val = min(r['min'] for r in ranges)
        max_val = max(r['max'] for r in ranges)
        return f"{min_val}-{max_val}"
    
    def create_vector_store(self, midi_files: List[str]):
        """MIDI 파일들의 특징을 벡터 스토어에 저장"""
        extractor = MIDIFeatureExtractor()
        texts = []
        
        for midi_file in midi_files:
            features = extractor.extract_features(midi_file)
            text = self.vectorize_features(features)
            texts.append(text)
        
        self.vector_store = Chroma.from_texts(
            texts,
            self.embeddings_model,
            metadatas=[{"source": f} for f in midi_files]
        )

class MIDIGenerator:
    """유사한 MIDI 파일을 기반으로 새로운 MIDI 생성"""
    def __init__(self, vectorizer: MIDIVectorizer, llm=None):
        self.vectorizer = vectorizer
        self.llm = llm or OpenAI(temperature=0.7)
        
        self.prompt_template = PromptTemplate(
            input_variables=["reference_features", "similar_features"],
            template="""
            입력된 MIDI 파일의 특징:
            {reference_features}
            
            유사한 MIDI 파일들의 특징:
            {similar_features}
            
            위 특징들을 기반으로 새로운 MIDI 파일을 생성하세요.
            출력 포맷은 다음과 같은 JSON 형식이어야 합니다:
            {
                "tracks": [
                    {
                        "instrument": <instrument_number>,
                        "notes": [
                            {"pitch": <midi_note>, "time": <start_time>, "duration": <duration>, "velocity": <velocity>},
                            ...
                        ]
                    },
                    ...
                ]
            }
            """
        )
        
    def generate(self, input_midi: str, n_similar=3) -> str:
        """입력 MIDI와 유사한 특징을 가진 새로운 MIDI 생성"""
        # 입력 MIDI의 특징 추출
        extractor = MIDIFeatureExtractor()
        input_features = extractor.extract_features(input_midi)
        input_text = self.vectorizer.vectorize_features(input_features)
        
        # 유사한 MIDI 찾기
        similar_docs = self.vectorizer.vector_store.similarity_search(input_text, k=n_similar)
        similar_text = "\n\n".join(doc.page_content for doc in similar_docs)
        
        # LLM으로 새로운 MIDI 생성
        chain = LLMChain(llm=self.llm, prompt=self.prompt_template)
        result = chain.run(
            reference_features=input_text,
            similar_features=similar_text
        )
        
        return result

def create_midi_from_json(json_data: str, output_file: str):
    """JSON 형식의 MIDI 데이터를 실제 MIDI 파일로 변환"""
    mid = mido.MidiFile()
    data = json.loads(json_data)
    
    for track_data in data['tracks']:
        track = mido.MidiTrack()
        track.append(mido.Message('program_change', program=track_data['instrument']))
        
        for note in track_data['notes']:
            # Note ON
            track.append(mido.Message('note_on', note=note['pitch'],
                                    velocity=note['velocity'],
                                    time=int(note['time'])))
            # Note OFF
            track.append(mido.Message('note_off', note=note['pitch'],
                                    velocity=0,
                                    time=int(note['duration'])))
        
        mid.tracks.append(track)
    
    mid.save(output_file)

# 사용 예시
def main():
    # 1. MIDI 파일들을 벡터화하여 저장
    vectorizer = MIDIVectorizer()
    midi_files = ["path/to/midi/files/*.mid"]  # MIDI 파일 경로 지정
    vectorizer.create_vector_store(midi_files)
    
    # 2. 생성기 초기화
    generator = MIDIGenerator(vectorizer)
    
    # 3. 새로운 MIDI 생성
    input_midi = "path/to/input.mid"  # 입력 MIDI 파일
    generated_json = generator.generate(input_midi)
    
    # 4. 생성된 MIDI 저장
    create_midi_from_json(generated_json, "output.mid")

if __name__ == "__main__":
    main()