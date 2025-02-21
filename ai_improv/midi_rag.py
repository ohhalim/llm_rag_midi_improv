from typing import List
import os
import json
from midi_feature_extractor import MIDIFeatureExtractor
from midi_vectorizer import MIDIVectorizer
from llm_api import LLMAPI

class MIDIRAGSystem:
    def __init__(self):
        self.vectorizer = MIDIVectorizer()
        self.llm_api = LLMAPI()
        self.feature_extractor = MIDIFeatureExtractor()
        
    def train(self, midi_files: List[str]):
        """MIDI 파일들로 RAG 시스템 학습"""
        self.vectorstore = self.vectorizer.vectorize_midi(midi_files)
    
    def generate(self, input_midi: str, output_format='json'):
        """
        입력 MIDI에 어울리는 새로운 MIDI 생성
        
        Args:
            input_midi (str): 입력 MIDI 파일 경로
            output_format (str): 출력 형식 ('json' 또는 'midi')
            
        Returns:
            str 또는 bytes: 'json' 형식이면 JSON 문자열, 'midi' 형식이면 MIDI 파일 바이트
        """
        # 입력 MIDI 특징 추출
        input_features = self.feature_extractor.extract_features(input_midi)
        
        # 유사한 MIDI 찾기
        similar_docs = self.vectorstore.similarity_search(str(input_features), k=3)
        similar_features = [doc.page_content for doc in similar_docs]
        
        # 새로운 MIDI 생성 (JSON 형식)
        json_response = self.llm_api.generate_response(
            str(input_features),
            "\n".join(similar_features)
        )
        
        if output_format == 'json':
            return json_response
        elif output_format == 'midi':
            # JSON을 MIDI로 변환
            from mido import MidiFile, MidiTrack, Message, MetaMessage
            
            try:
                # JSON 파싱
                data = json.loads(json_response)
                
                # MIDI 파일 생성
                mid = MidiFile()
                
                # 템포 트랙 추가
                tempo_track = MidiTrack()
                mid.tracks.append(tempo_track)
                
                # 템포 설정 (120 BPM)
                tempo_track.append(MetaMessage('set_tempo', tempo=500000, time=0))
                
                # 타임 시그니처 설정
                time_signatures = data.get('time_signatures', ['4/4'])
                if time_signatures and len(time_signatures) > 0:
                    time_sig = time_signatures[0].split('/')
                    if len(time_sig) == 2:
                        numerator, denominator = int(time_sig[0]), int(time_sig[1])
                        tempo_track.append(MetaMessage('time_signature', 
                                                      numerator=numerator, 
                                                      denominator=denominator,
                                                      clocks_per_click=24,
                                                      notated_32nd_notes_per_beat=8,
                                                      time=0))
                
                # 키 시그니처 설정 (있는 경우)
                key_signatures = data.get('key_signatures', [])
                if key_signatures and len(key_signatures) > 0:
                    tempo_track.append(MetaMessage('key_signature', key=key_signatures[0], time=0))
                
                # 트랙 추가
                for track_data in data.get('tracks', []):
                    track = MidiTrack()
                    mid.tracks.append(track)
                    
                    # 악기 설정
                    instrument = track_data.get('instrument', 0)
                    track.append(Message('program_change', program=instrument, time=0))
                    
                    # 노트 추가
                    notes = track_data.get('notes', [])
                    notes.sort(key=lambda x: x.get('time', 0))  # 시간 순으로 정렬
                    
                    def seconds_to_ticks(seconds):
                        """초 단위 시간을 MIDI 틱으로 변환"""
                        # 기본값: 480 ticks per quarter note, 120 BPM (500000 microseconds per beat)
                        return int(seconds * 480 * 120 / 60)
                    
                    last_time = 0
                    for note in notes:
                        pitch = note.get('pitch', 60)  # 기본값: 중간 C
                        time_seconds = note.get('time', 0)
                        duration_seconds = note.get('duration', 1)
                        velocity = note.get('velocity', 64)
                        
                        # 시간을 틱으로 변환
                        time_ticks = seconds_to_ticks(time_seconds)
                        duration_ticks = seconds_to_ticks(duration_seconds)
                        
                        # 이전 이벤트로부터의 상대적 시간
                        delta_ticks = time_ticks - last_time
                        
                        # 노트 온
                        track.append(Message('note_on', note=pitch, velocity=velocity, time=max(0, delta_ticks)))
                        
                        # 노트 오프 (노트 온 후에 duration_ticks 시간 후)
                        track.append(Message('note_off', note=pitch, velocity=0, time=duration_ticks))
                        
                        # 마지막 이벤트 시간 업데이트
                        last_time = time_ticks + duration_ticks
                
                # 메모리에 MIDI 파일 생성
                import io
                buffer = io.BytesIO()
                mid.save(file=buffer)
                buffer.seek(0)
                return buffer.read()
                
            except Exception as e:
                print(f"MIDI 변환 중 오류 발생: {str(e)}")
                return json_response  # 오류 발생 시 JSON 반환
        else:
            raise ValueError(f"지원하지 않는 출력 형식: {output_format}")

    def save_midi(self, midi_data, output_path):
        """생성된 MIDI 데이터를 파일로 저장"""
        if isinstance(midi_data, str):
            # JSON 형식인 경우
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(midi_data)
        else:
            # MIDI 바이너리 데이터인 경우
            with open(output_path, 'wb') as f:
                f.write(midi_data)