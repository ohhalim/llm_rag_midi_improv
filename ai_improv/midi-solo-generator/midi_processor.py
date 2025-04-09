import os
import mido
import json
import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Dict, List, Optional, Any, Tuple
import uuid
from datetime import datetime
import numpy as np
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.retrievers import ContextualCompressionRetriever
import io



class PositionalEncoding(nn.Module):
    """
    트랜스포머를 위한 위치 인코딩
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class MusicTransformerModel(nn.Module):
    """
    PyTorch 기반 Music Transformer 모델
    """
    def __init__(self, d_model=256, nhead=8, num_encoder_layers=6, 
                 num_decoder_layers=6, dim_feedforward=1024, dropout=0.1):
        super(MusicTransformerModel, self).__init__()
        
        # Transformer 모델 구성
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        
        # MIDI 이벤트 토큰화를 위한 임베딩 레이어
        self.event_embedding = nn.Embedding(128 + 128 + 100, d_model)  # note_on + note_off + time_shifts
        
        # 출력 레이어
        self.output_layer = nn.Linear(d_model, 128 + 128 + 100)
        
        # 위치 인코딩
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
    def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None):
        # 임베딩 및 위치 인코딩
        src = self.event_embedding(src) * np.sqrt(self.transformer.d_model)
        src = self.pos_encoder(src)
        
        tgt = self.event_embedding(tgt) * np.sqrt(self.transformer.d_model)
        tgt = self.pos_encoder(tgt)
        
        # Transformer 모델 통과
        output = self.transformer(src, tgt, src_mask, tgt_mask, None, None, None, memory_mask)
        
        # 출력 레이어
        output = self.output_layer(output)
        return output


class MusicTransformerProcessor:
    """
    Music Transformer 모델을 사용하여 MIDI 데이터 처리
    """
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model_name = os.path.basename(model_path)
        self.loaded_at = datetime.now().isoformat()
        
        # 모델 설정
        self.metadata = {
            "name": "Music Transformer Solo Generator",
            "version": "1.0.0",
            "description": "MIDI 솔로 라인 생성을 위한 Music Transformer 모델",
            "architecture": "Transformer (PyTorch)",
            "input_type": "MIDI",
            "output_type": "MIDI",
            "supported_styles": ["jazz_funk", "bebop", "blues", "fusion"]
        }
        
        # MIDI 이벤트 토큰화 설정
        self.note_range = 128  # MIDI 노트 범위 (0-127)
        self.velocity_bins = 32  # 벨로시티 양자화 빈
        self.time_bins = 100  # 타임 시프트 양자화 빈
        
        # 디바이스 설정
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # Music Transformer 모델 로드
        self._load_model()
    
    def _load_model(self):
        """Music Transformer 모델 로드"""
        try:
            # 모델 초기화
            self.model = MusicTransformerModel()
            
            # 사전 훈련된 모델 가중치 로드 (모델 파일이 존재하면)
            if os.path.exists(self.model_path):
                self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
                print(f"Music Transformer 모델 로드 완료: {self.model_name}")
            else:
                print(f"경고: 모델 파일 {self.model_path}가 존재하지 않음, 가중치가 초기화된 모델 사용")
            
            # 모델을 추론 모드로 변환
            self.model.eval()
            self.model.to(self.device)
            
        except Exception as e:
            print(f"모델 로드 오류: {str(e)}")
            self.model = None
            print("경고: 모델 로드에 실패했습니다. 가상 모델을 사용합니다.")
    
    def process_input(self, midi_data: bytes) -> Dict:
        """
        Music Transformer를 통해 입력 MIDI 데이터 처리
        
        Args:
            midi_data: 바이트로 된 raw MIDI 데이터
            
        Returns:
            처리된 컨텍스트가 포함된 Dict
        """
        try:
            # MIDI 데이터 파싱
            midi_file = mido.MidiFile(file=io.BytesIO(midi_data))
            
            # MIDI 노트 및 특성 추출
            notes, tempo = self._extract_notes_and_tempo(midi_file)
            
            # 음악적 특성 분석
            pitch_histogram = self._calculate_pitch_histogram(notes)
            key = self._infer_key(pitch_histogram)
            chord_progression = self._infer_chord_progression(notes, key)
            
            # MIDI를 토큰화하여 PyTorch 텐서로 변환
            tokens = self._tokenize_midi(midi_file)
            
            # 컨텍스트 생성
            context = {
                "midi_info": {
                    "num_notes": len(notes),
                    "duration": notes[-1]['time'] + notes[-1]['duration'] if notes else 0
                },
                "tokens": tokens.tolist() if tokens is not None else [],
                "yue_features": {  # 기존 YuE 호환성 유지
                    "harmonic_context": {
                        "key": key,
                        "chord_progression": chord_progression
                    },
                    "melodic_features": {
                        "pitch_histogram": pitch_histogram,
                        "contour": self._analyze_contour(notes),
                        "range": self._calculate_range(notes)
                    },
                    "rhythmic_features": {
                        "tempo": tempo,
                        "density": len(notes) / (notes[-1]['time'] + notes[-1]['duration']) if notes else 0,
                        "syncopation": self._calculate_syncopation(notes, tempo)
                    }
                },
                "transformer_data": {
                    "model_name": self.model_name,
                }
            }
            
            return context
            
        except Exception as e:
            print(f"MIDI 처리 오류: {str(e)}")
            return {
                "error": str(e),
                "yue_features": {
                    "harmonic_context": {
                        "chord_progression": ["Cmaj7", "Am7", "Dm7", "G7"]  # 기본값
                    },
                    "rhythmic_features": {
                        "tempo": 120
                    }
                }
            }
    
    def _extract_notes_and_tempo(self, midi_file: mido.MidiFile) -> Tuple[List[Dict], float]:
        """MIDI 파일에서 노트 정보와 템포 추출"""
        notes = []
        tempo = 120.0  # 기본 템포
        
        for track in midi_file.tracks:
            time_in_ticks = 0
            active_notes = {}
            
            for msg in track:
                time_in_ticks += msg.time
                
                if msg.type == 'set_tempo':
                    tempo = mido.tempo2bpm(msg.tempo)
                
                elif msg.type == 'note_on' and msg.velocity > 0:
                    # 노트 시작
                    active_notes[msg.note] = {
                        "pitch": msg.note,
                        "time": time_in_ticks / midi_file.ticks_per_beat * (60 / tempo),
                        "velocity": msg.velocity
                    }
                
                elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                    # 노트 종료
                    if msg.note in active_notes:
                        note_data = active_notes[msg.note]
                        current_time = time_in_ticks / midi_file.ticks_per_beat * (60 / tempo)
                        note_data["duration"] = current_time - note_data["time"]
                        notes.append(note_data)
                        del active_notes[msg.note]
        
        # 시간순으로 정렬
        notes.sort(key=lambda x: x["time"])
        return notes, tempo
    
    def _tokenize_midi(self, midi_file: mido.MidiFile) -> torch.Tensor:
        """MIDI 파일을 토큰화하여 PyTorch 텐서로 변환"""
        tokens = []
        
        for track in midi_file.tracks:
            time_in_ticks = 0
            previous_tick = 0
            
            for msg in track:
                time_in_ticks += msg.time
                
                # 시간 토큰 추가 (델타 시간)
                if msg.time > 0:
                    # 델타 시간을 양자화
                    delta_time = time_in_ticks - previous_tick
                    time_token = self._quantize_time(delta_time, midi_file.ticks_per_beat)
                    tokens.append(2*self.note_range + time_token)  # 시간 토큰은 노트 토큰 다음에 위치
                
                # 노트 토큰 추가
                if msg.type == 'note_on' and msg.velocity > 0:
                    # 노트 온 토큰
                    tokens.append(msg.note)  # 노트 온 토큰 (0-127)
                    previous_tick = time_in_ticks
                
                elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                    # 노트 오프 토큰
                    tokens.append(self.note_range + msg.note)  # 노트 오프 토큰 (128-255)
                    previous_tick = time_in_ticks
        
        # PyTorch 텐서로 변환
        if tokens:
            return torch.tensor(tokens, dtype=torch.long)
        else:
            return None
    
    def _quantize_time(self, delta_time: int, ticks_per_beat: int) -> int:
        """델타 시간을 양자화"""
        # 비트 단위로 변환
        beats = delta_time / ticks_per_beat
        
        # 양자화 (0.0부터 2.0 비트까지 time_bins 단계로 분할)
        max_beat = 2.0
        bin_idx = min(int(beats / max_beat * self.time_bins), self.time_bins - 1)
        return bin_idx
    
    def _calculate_pitch_histogram(self, notes: List[Dict]) -> List[float]:
        """피치 클래스 분포 계산"""
        histogram = [0] * 12
        
        for note in notes:
            pitch_class = note["pitch"] % 12
            histogram[pitch_class] += 1
        
        # 정규화
        total = sum(histogram)
        if total > 0:
            histogram = [h / total for h in histogram]
        
        return histogram
    
    def _infer_key(self, pitch_histogram: List[float]) -> str:
        """피치 히스토그램에서 키 추론"""
        # 간단한 키 감지 알고리즘
        key_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        
        # 메이저와 마이너 키 프로파일
        major_profile = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
        minor_profile = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
        
        # 상관관계 계산
        major_corr = []
        minor_corr = []
        
        for i in range(12):  # 모든 가능한 키
            major_sum = 0
            minor_sum = 0
            
            for j in range(12):  # 각 피치 클래스
                major_sum += pitch_histogram[j] * major_profile[(j - i) % 12]
                minor_sum += pitch_histogram[j] * minor_profile[(j - i) % 12]
            
            major_corr.append(major_sum)
            minor_corr.append(minor_sum)
        
        # 최대 상관관계 찾기
        max_major = max(major_corr)
        max_minor = max(minor_corr)
        
        if max_major > max_minor:
            # 메이저 키
            key_idx = major_corr.index(max_major)
            return key_names[key_idx]
        else:
            # 마이너 키
            key_idx = minor_corr.index(max_minor)
            return key_names[key_idx] + "m"
    
    def _infer_chord_progression(self, notes: List[Dict], key: str) -> List[str]:
        """노트 데이터에서 코드 진행 추론"""
        # 키에서 코드 진행 유추
        is_minor = key.endswith("m")
        root_note = key[0] if not is_minor else key[0:-1]
        
        note_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        root_idx = note_names.index(root_note)
        
        # 시간 구간 나누기
        if notes:
            total_time = notes[-1]["time"] + notes[-1]["duration"]
            num_segments = min(4, max(1, int(total_time / 2)))  # 2초마다 하나의 코드 가정
        else:
            num_segments = 4
        
        if is_minor:
            # 마이너 키 일반적 진행
            chord_types = ["m7", "m7", "maj7", "7"]
            intervals = [0, 5, 7, 2]  # i, iv, V, ii 진행
        else:
            # 메이저 키 일반적 진행
            chord_types = ["maj7", "m7", "m7", "7"]
            intervals = [0, 2, 5, 7]  # I, ii, IV, V 진행
        
        # 코드 진행 생성
        chord_progression = []
        for i in range(num_segments):
            chord_idx = i % len(intervals)
            note_idx = (root_idx + intervals[chord_idx]) % 12
            chord = note_names[note_idx] + chord_types[chord_idx]
            chord_progression.append(chord)
        
        return chord_progression
    
    def _analyze_contour(self, notes: List[Dict]) -> str:
        """멜로디 윤곽 분석"""
        if not notes:
            return "flat"
            
        # 첫 번째와 마지막 음표의 피치 비교
        first_pitch = notes[0]["pitch"]
        last_pitch = notes[-1]["pitch"]
        
        if last_pitch > first_pitch + 3:
            return "ascending"
        elif last_pitch < first_pitch - 3:
            return "descending"
        else:
            # 피치 변화 추세 계산
            pitches = [n["pitch"] for n in notes]
            changes = [pitches[i+1] - pitches[i] for i in range(len(pitches)-1)]
            
            # 변화 방향 분석
            up_count = sum(1 for c in changes if c > 0)
            down_count = sum(1 for c in changes if c < 0)
            
            if up_count > down_count * 2:
                return "ascending"
            elif down_count > up_count * 2:
                return "descending"
            elif up_count > len(changes) * 0.4 and down_count > len(changes) * 0.4:
                return "undulating"
            else:
                return "arch"
    
    def _calculate_range(self, notes: List[Dict]) -> int:
        """음표 범위 계산 (반음 단위)"""
        if not notes:
            return 0
            
        pitches = [note["pitch"] for note in notes]
        return max(pitches) - min(pitches)
    
    def _calculate_syncopation(self, notes: List[Dict], tempo: float) -> float:
        """싱코페이션 수준 계산"""
        if not notes:
            return 0.0
        
        # 비트 길이 (초 단위)
        beat_duration = 60 / tempo
        
        # 오프비트 감지
        off_beat_count = 0
        for note in notes:
            # 노트 시작 시간을 비트 단위로 변환한 후 비트 내 위치 계산
            beat_position = (note["time"] / beat_duration) % 1.0
            
            # 오프비트 범위 정의 (비트의 중간 부근)
            if 0.25 < beat_position < 0.4 or 0.6 < beat_position < 0.75:
                off_beat_count += 1
        
        return off_beat_count / len(notes) if notes else 0.0
    
    def get_model_info(self) -> Dict:
        """현재 로드된 Music Transformer 모델에 대한 정보 가져오기"""
        return {
            "name": self.metadata["name"],
            "path": self.model_path,
            "loaded_at": self.loaded_at,
            "model_status": "active" if self.model is not None else "inactive",
            "device": str(self.device)
        }
    
    def generate(self, tokens: torch.Tensor, length: int = 100, temperature: float = 1.0) -> torch.Tensor:
        """
        입력 토큰에 기반한 새로운 토큰 시퀀스 생성
        
        Args:
            tokens: 입력 토큰 시퀀스
            length: 생성할 토큰 수
            temperature: 샘플링 온도 (높을수록 더 무작위적)
            
        Returns:
            생성된 토큰 시퀀스
        """
        if self.model is None:
            # 모델이 없는 경우 무작위 토큰 생성
            return torch.randint(0, 2*self.note_range + self.time_bins, (length,))
        
        # 모델을 추론 모드로 전환
        self.model.eval()
        
        # 디바이스로 입력 이동
        input_tensor = tokens.to(self.device)
        
        # 결과 저장을 위한 출력 텐서
        output_tokens = tokens.clone()
        
        with torch.no_grad():
            for i in range(length):
                # 현재까지의 입력으로 다음 토큰 예측
                src = input_tensor.unsqueeze(1)
                tgt = output_tokens[-1].unsqueeze(0).unsqueeze(1)
                
                # 소스 및 타겟 마스크 생성
                src_mask = torch.zeros((src.size(0), src.size(0)), device=self.device).type(torch.bool)
                tgt_mask = torch.zeros((1, 1), device=self.device).type(torch.bool)
                
                # 모델을 통한 예측
                output = self.model(src, tgt, src_mask, tgt_mask)
                output = output.squeeze(0)
                
                # 온도 적용 및 다음 토큰 샘플링
                output = output / temperature
                probs = F.softmax(output, dim=-1)
                next_token = torch.multinomial(probs, 1).item()
                
                # 결과에 추가
                output_tokens = torch.cat([output_tokens, torch.tensor([next_token])])
        
        return output_tokens


class MIDIGenerator:
    """
    Music Transformer를 사용하여 MIDI 솔로 생성
    """
    def __init__(self, midi_db_path: str):
        self.midi_db_path = midi_db_path
        self.embeddings = None
        self.db = None
        self.retriever = None
        
        # LangChain 컴포넌트 설정
        self._setup_langchain()
    
    def _setup_langchain(self):
        """LangChain 컴포넌트 설정"""
        try:
            # MIDI용 임베딩 초기화
            self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            
            # 디렉토리 생성 확인
            os.makedirs(self.midi_db_path, exist_ok=True)
            
            # MIDI 데이터베이스의 벡터 저장소 생성
            self.db = Chroma(
                persist_directory=self.midi_db_path,
                embedding_function=self.embeddings
            )
            
            # RAG 검색기 설정
            self.retriever = ContextualCompressionRetriever(
                base_compressor=self.db.as_retriever(search_kwargs={"k": 5})
            )
        except Exception as e:
            print(f"LangChain 설정 오류: {str(e)}")
            # 오류 발생 시 모의 구성으로 폴백
            self.db = {
                "status": "mock_initialized",
                "db_path": self.midi_db_path,
                "initialized_at": datetime.now().isoformat()
            }
    
    def generate_solo(self, context: Dict, style: str = "jazz_funk", complexity: float = 0.7, length: int = 32) -> bytes:
        """
        컨텍스트와 스타일에 기반한 새로운 즉흥 솔로 라인 생성
        
        Args:
            context: 음악 컨텍스트 딕셔너리
            style: 즉흥연주 스타일
            complexity: 복잡도 (0.0-1.0)
            length: 솔로 라인 길이 (마디 수)
            
        Returns:
            바이트로 된 MIDI 데이터
        """
        try:
            # 트랜스포머 토큰 확인
            if "tokens" in context and len(context["tokens"]) > 0:
                # PyTorch 텐서로 변환
                input_tokens = torch.tensor(context["tokens"])
                
                # 온도 설정 (복잡도에 따라)
                temperature = 0.5 + (complexity * 0.5)  # 0.5 ~ 1.0
                
                # 생성할 토큰 수 계산 (마디당 약 16개 토큰 가정)
                num_tokens = length * 16
                
                # 모델을 통한 생성
                output_tokens = context["transformer"].generate(
                    input_tokens, 
                    length=num_tokens, 
                    temperature=temperature
                )
                
                # 토큰을 MIDI로 변환
                output_midi = self._tokens_to_midi(output_tokens, style)
                return output_midi
            else:
                # 토큰이 없는 경우 스타일 기반 생성으로 폴백
                print("Music Transformer 토큰 없음, 스타일 기반 생성으로 폴백")
                return self._generate_style_based_solo(context, style, complexity, length)
                
        except Exception as e:
            print(f"솔로 생성 오류: {str(e)}")
            # 오류 시 스타일 기반 생성으로 폴백
            return self._generate_style_based_solo(context, style, complexity, length)
    
    def _tokens_to_midi(self, tokens: torch.Tensor, style: str) -> bytes:
        """토큰 시퀀스를 MIDI 바이트로 변환"""
        # 새 MIDI 파일 생성
        mid = mido.MidiFile()
        track = mido.MidiTrack()
        mid.tracks.append(track)
        
        # 트랙 설정 - 스타일에 따른 악기 선택
        if style == "jazz_funk":
            instrument = 66  # 테너 색소폰
        elif style == "bebop":
            instrument = 65  # 알토 색소폰
        elif style == "blues":
            instrument = 26  # 일렉트릭 기타
        else:
            instrument = 66  # 기본: 테너 색소폰
        
        track.append(mido.Message('program_change', program=instrument, time=0))
        
        # 템포 설정
        track.append(mido.MetaMessage('set_tempo', tempo=mido.bpm2tempo(120), time=0))
        
        # 노트 온/오프 활성 상태 추적
        active_notes = set()
        current_time = 0
        
        note_range = 128
        time_bins = 100
        
        for token in tokens:
            token = token.item()
            
            if token < note_range:
                # 노트 온 토큰
                note = token
                track.append(mido.Message('note_on', note=note, velocity=80, time=current_time))
                active_notes.add(note)
                current_time = 0
            
            elif token < 2 * note_range:
                # 노트 오프 토큰
                note = token - note_range
                if note in active_notes:
                    track.append(mido.Message('note_off', note=note, velocity=0, time=current_time))
                    active_notes.remove(note)
                    current_time = 0
            
            else:
                # 시간 토큰
                time_bin = token - 2 * note_range
                # 시간 빈을 틱으로 변환 (0.01초 ~ 2초 범위)
                time_in_seconds = time_bin / time_bins * 2.0
                time_in_ticks = int(time_in_seconds * 480)  # 480 ticks per beat
                current_time += time_in_ticks
        
        # 남아있는 노트 종료
        for note in active_notes:
            track.append(mido.Message('note_off', note=note, velocity=0, time=current_time))
        
        # BytesIO 객체로 저장하고 바이트 반환
        buffer = io.BytesIO()
        mid.save(file=buffer)
        buffer.seek(0)
        return buffer.read()
    
    def _generate_style_based_solo(self, context: Dict, style: str, complexity: float, length: int) -> bytes:
        """
        스타일 기반 MIDI 솔로 생성 (Music Transformer 사용 불가 시 폴백)
        """
        # 새 MIDI 파일 생성
        mid = mido.MidiFile()
        track = mido.MidiTrack()
        mid.tracks.append(track)
        
        # 트랙 및 템포 설정
        if style == "jazz_funk":
            track.append(mido.Message('program_change', program=66, time=0))  # 테너 색소폰
        elif style == "bebop":
            track.append(mido.Message('program_change', program=65, time=0))  # 알토 색소폰
        elif style == "blues":
            track.append(mido.Message('program_change', program=26, time=0))  # 일렉트릭 기타
        else:
            track.append(mido.Message('program_change', program=66, time=0))  # 테너 색소폰 (기본값)
        
        # 컨텍스트에서 템포 가져오기
        tempo = 120  # 기본값
        if ("yue_features" in context and 
            "rhythmic_features" in context["yue_features"] and 
            "tempo" in context["yue_features"]["rhythmic_features"]):
            tempo = context["yue_features"]["rhythmic_features"]["tempo"]
        
        # 템포 설정
        track.append(mido.MetaMessage('set_tempo', tempo=mido.bpm2tempo(tempo), time=0))
        
        # 코드 진행 가져오기
        chord_progression = ["Cmaj7", "Am7", "Dm7", "G7"]  # 기본값
        if ("yue_features" in context and 
            "harmonic_context" in context["yue_features"] and 
            "chord_progression" in context["yue_features"]["harmonic_context"]):
            chord_progression = context["yue_features"]["harmonic_context"]["chord_progression"]
        
        # 스타일별 노트 생성
        if style == "jazz_funk":
            notes = self._create_jazz_funk_notes(chord_progression, complexity, length)
        elif style == "bebop":
            notes = self._create_bebop_notes(chord_progression, complexity, length)
        elif style == "blues":
            notes = self._create_blues_notes(chord_progression, complexity, length)
        else:
            notes = self._create_jazz_funk_notes(chord_progression, complexity, length)
        
        # 노트를 MIDI 메시지로 변환
        prev_time = 0
        for note in notes:
            # Note on
            track.append(mido.Message('note_on', note=note['pitch'], 
                                     velocity=note['velocity'], 
                                     time=int(480 * (note['time'] - prev_time))))
            prev_time = note['time']
            
            # Note off
            track.append(mido.Message('note_off', note=note['pitch'], 
                                     velocity=0, 
                                     time=int(480 * note['duration'])))
            prev_time += note['duration']
        
        # BytesIO 객체로 저장하고 바이트 반환
        buffer = io.BytesIO()
        mid.save(file=buffer)
        buffer.seek(0)
        return buffer.read()
    
    def _create_jazz_funk_notes(self, chord_progression: List[str], complexity: float, length: int) -> List[Dict]:
        """코드 진행에서 재즈 펑크 스타일의 노트 생성"""
        notes = []
        
        # 루트 음표 사전
        root_notes = {
            "C": 60, "C#": 61, "Db": 61, "D": 62, "D#": 63, "Eb": 63,
            "E": 64, "F": 65, "F#": 66, "Gb": 66, "G": 67, "G#": 68,
            "Ab": 68, "A": 69, "A#": 70, "Bb": 70, "B": 71
        }
        
        # 코드 유형별 음정
        chord_types = {
            "maj": [0, 4, 7],
            "min": [0, 3, 7],
            "7": [0, 4, 7, 10],
            "maj7": [0, 4, 7, 11],
            "m7": [0, 3, 7, 10],
            "dim": [0, 3, 6],
            "aug": [0, 4, 8],
            "sus4": [0, 5, 7]
        }
        
        # 시간 카운터
        time = 0.0
        
        # 각 코드에 대해 노트 생성
        for _ in range(length):  # 지정된 길이만큼 반복
            for chord in chord_progression:
                # 코드 파싱
                root = ""
                chord_type = ""
                
                # 코드 루트 추출
                for i, char in enumerate(chord):
                    if char.isalpha() or char in ['#', 'b']:
                        root += char
                    else:
                        chord_type = chord[i:]
                        break
                        
                if not chord_type:
                    chord_type = "maj"  # 기본값
                
                # 코드 타입 정규화
                if chord_type == "m":
                    chord_type = "min"
                elif chord_type == "m7":
                    chord_type = "m7"
                elif chord_type == "M7":
                    chord_type = "maj7"
                
                # 루트 노트 가져오기
                if root in root_notes:
                    root_note = root_notes[root]
                else:
                    root_note = 60  # 기본값 C
                
                # 코드 음표 가져오기
                intervals = chord_types.get(chord_type, chord_types["maj"])
                chord_notes = [root_note + i for i in intervals]
                
                # 펑크 스타일의 리듬적 요소 추가
                if np.random.random() < complexity:
                    # 16분음표 패턴 (높은 복잡도)
                    rhythm_pattern = [0.25, 0.25, 0.25, 0.25]
                else:
                    # 8분음표 패턴 (낮은 복잡도)
                    rhythm_pattern = [0.5, 0.5]
                
                # 코드 음표에서 선택
                for r in rhythm_pattern:
                    # 복잡도에 따른 코드 외 음표 선택 확률
                    if np.random.random() < complexity * 0.7:
                        # 코드 음표 외의 음계 음표 선택
                        pitch = root_note + np.random.choice([2, 5, 9, 12, 14])
                    else:
                        # 코드 음표 선택
                        pitch = np.random.choice(chord_notes)
                    
                    # 랜덤 옥타브 변경
                    if np.random.random() < 0.3:
                        pitch += 12  # 한 옥타브 위
                    
                    # 벨로시티 랜덤화 (강약)
                    velocity = np.random.randint(70, 110)
                    
                    # 노트 추가
                    notes.append({
                        "pitch": pitch,
                        "time": time,
                        "duration": r * 0.9,  # 약간의 스타카토 효과
                        "velocity": velocity
                    })
                    
                    time += r
        
        return notes
    
    def _create_bebop_notes(self, chord_progression: List[str], complexity: float, length: int) -> List[Dict]:
        """코드 진행에서 비밥 스타일의 노트 생성"""
        notes = []
        
        # 루트 음표 사전
        root_notes = {
            "C": 60, "C#": 61, "Db": 61, "D": 62, "D#": 63, "Eb": 63,
            "E": 64, "F": 65, "F#": 66, "Gb": 66, "G": 67, "G#": 68,
            "Ab": 68, "A": 69, "A#": 70, "Bb": 70, "B": 71
        }
        
        # 비밥 스케일 (코드 음표 + 패싱 톤)
        bebop_scales = {
            "maj7": [0, 2, 4, 5, 7, 9, 11, 12],     # 메이저 비밥 스케일
            "7": [0, 2, 4, 5, 7, 9, 10, 11, 12],    # 도미넌트 비밥 스케일
            "m7": [0, 2, 3, 5, 7, 9, 10, 12],       # 마이너 비밥 스케일
            "dim": [0, 1, 3, 4, 6, 7, 9, 10, 12]    # 디미니쉬드 비밥 스케일
        }
        
        # 시간 카운터
        time = 0.0
        
        # 각 코드에 대해 노트 생성
        for _ in range(length):  # 지정된 길이만큼 반복
            for chord in chord_progression:
                # 코드 파싱
                root = ""
                chord_type = ""
                
                # 코드 루트 추출
                for i, char in enumerate(chord):
                    if char.isalpha() or char in ['#', 'b']:
                        root += char
                    else:
                        chord_type = chord[i:]
                        break
                        
                if not chord_type:
                    chord_type = "maj"  # 기본값
                
                # 코드 타입 정규화
                if chord_type == "m":
                    chord_type = "min"
                elif chord_type == "m7":
                    chord_type = "m7"
                elif chord_type == "M7":
                    chord_type = "maj7"
                
                # 루트 노트 가져오기
                if root in root_notes:
                    root_note = root_notes[root]
                else:
                    root_note = 60  # 기본값 C
                
                # 비밥 스케일 선택
                if chord_type in bebop_scales:
                    scale_intervals = bebop_scales[chord_type]
                elif "7" in chord_type:
                    scale_intervals = bebop_scales["7"]
                elif "m" in chord_type:
                    scale_intervals = bebop_scales["m7"]
                else:
                    scale_intervals = bebop_scales["maj7"]
                
                scale_notes = [root_note + i for i in scale_intervals]
                
                # 비밥 스타일의 리듬적 요소 추가 (빠른 16분음표)
                if np.random.random() < complexity:
                    # 복잡한 비밥 패턴
                    rhythm_pattern = [0.125, 0.125, 0.125, 0.125, 0.25, 0.25]
                else:
                    # 단순한 비밥 패턴
                    rhythm_pattern = [0.25, 0.25, 0.25, 0.25]
                
                # 비밥 스타일 라인 생성
                current_idx = np.random.randint(0, len(scale_notes))
                for r in rhythm_pattern:
                    # 비밥 특유의 스케일 순차 진행 (상승 또는 하강)
                    direction = 1 if np.random.random() > 0.5 else -1
                    
                    # 간혹 큰 도약도 사용
                    if np.random.random() < complexity * 0.2:
                        jump = np.random.choice([3, 4, 7])
                        current_idx = (current_idx + direction * jump) % len(scale_notes)
                    else:
                        current_idx = (current_idx + direction) % len(scale_notes)
                    
                    pitch = scale_notes[current_idx]
                    
                    # 랜덤 옥타브 변경
                    if np.random.random() < 0.1:
                        pitch += 12  # 한 옥타브 위
                    
                    # 벨로시티 랜덤화 (비밥 특유의 악센트 표현)
                    velocity = np.random.randint(80, 127) if np.random.random() < 0.3 else np.random.randint(70, 100)
                    
                    # 노트 추가
                    notes.append({
                        "pitch": pitch,
                        "time": time,
                        "duration": r * 0.8,  # 비밥 특유의 강한 스타카토
                        "velocity": velocity
                    })
                    
                    time += r
        
        return notes
    
    def _create_blues_notes(self, chord_progression: List[str], complexity: float, length: int) -> List[Dict]:
        """코드 진행에서 블루스 스타일의 노트 생성"""
        notes = []
        
        # 루트 음표 사전
        root_notes = {
            "C": 60, "C#": 61, "Db": 61, "D": 62, "D#": 63, "Eb": 63,
            "E": 64, "F": 65, "F#": 66, "Gb": 66, "G": 67, "G#": 68,
            "Ab": 68, "A": 69, "A#": 70, "Bb": 70, "B": 71
        }
        
        # 블루스 스케일 구성
        blues_scale_intervals = [0, 3, 5, 6, 7, 10, 12]  # 마이너 블루스 스케일
        
        # 시간 카운터
        time = 0.0
        
        # 첫 번째 코드의 루트 음을 기준으로 블루스 스케일 생성
        first_chord = chord_progression[0]
        root = ""
        for char in first_chord:
            if char.isalpha() or char in ['#', 'b']:
                root += char
            else:
                break
        
        # 루트 노트 가져오기
        if root in root_notes:
            root_note = root_notes[root]
        else:
            root_note = 60  # 기본값 C
        
        # 블루스 스케일 노트
        blues_notes = [root_note + i for i in blues_scale_intervals]
        
        # 각 코드에 대해 노트 생성
        for _ in range(length):  # 지정된 길이만큼 반복
            for chord in chord_progression:
                # 블루스 패턴의 리듬적 요소 추가
                if np.random.random() < complexity:
                    # 복잡한 블루스 패턴 (트리플렛 포함)
                    rhythm_pattern = [0.33, 0.33, 0.33, 0.5, 0.5]
                else:
                    # 단순한 블루스 패턴
                    rhythm_pattern = [0.5, 0.5, 1.0]
                
                # 블루스 스타일 라인 생성
                for r in rhythm_pattern:
                    # 블루스 특유의 벤딩 노트 효과
                    use_bend = np.random.random() < complexity * 0.4
                    
                    # 블루스 음표 선택
                    pitch_idx = np.random.randint(0, len(blues_notes))
                    pitch = blues_notes[pitch_idx]
                    
                    # 블루 노트 (b3, b5, b7) 강조
                    if np.random.random() < 0.6:
                        pitch = root_note + np.random.choice([3, 6, 10])
                    
                    # 벨로시티 랜덤화 (블루스 특유의 강약)
                    velocity = np.random.randint(70, 110)
                    
                    # 블루스 특유의 표현 추가
                    if use_bend:
                        # 노트 추가 (밴드 업)
                        notes.append({
                            "pitch": pitch - 1,  # 반음 아래에서 시작
                            "time": time,
                            "duration": r * 0.3,
                            "velocity": velocity
                        })
                        
                        # 벤드 업된 노트
                        notes.append({
                            "pitch": pitch,
                            "time": time + r * 0.3,
                            "duration": r * 0.7,
                            "velocity": velocity
                        })
                    else:
                        # 일반 노트 추가
                        notes.append({
                            "pitch": pitch,
                            "time": time,
                            "duration": r * 0.9,  # 약간의 스타카토 효과
                            "velocity": velocity
                        })
                    
                    time += r
        
        return notes
    
    def get_status(self) -> Dict:
        """MIDI 생성기의 상태 가져오기"""
        if isinstance(self.db, dict):
            return self.db
        
        return {
            "status": "active",
            "db_path": self.midi_db_path,
            "retriever_type": "ContextualCompressionRetriever",
            "embedding_model": "all-MiniLM-L6-v2"
        }


class FLStudioInterface:
    """
    MCP(MIDI Control Protocol)를 사용하여 FL Studio와 인터페이스
    """
    def __init__(self, port: int = 8080):
        self.port = port
        self.connected = False
        self.connection_info = {
            "port": port,
            "status": "disconnected",
            "connected_at": None
        }
        
        # 실제 구현에서는 FL Studio에 MCP를 통해 연결
        # 이 예시에서는 연결을 시뮬레이션
        self._connect()
    
    def _connect(self):
        """FL Studio에 연결 시뮬레이션"""
        # 실제 구현에서는 실제로 FL Studio에 연결
        self.connected = True
        self.connection_info["status"] = "connected"
        self.connection_info["connected_at"] = datetime.now().isoformat()
    
    def send_midi(self, midi_data: bytes) -> bool:
        """
        MIDI 데이터를 FL Studio로 전송
        
        Args:
            midi_data: 바이트로 된 raw MIDI 데이터
            
        Returns:
            성공 여부를 나타내는 Boolean
        """
        if not self.connected:
            return False
        
        # 실제 구현에서는 MCP를 통해 FL Studio로 MIDI 데이터 전송
        # 이 예시에서는 성공했다고 가정
        
        return True
    
    def get_connection_status(self) -> Dict:
        """현재 연결 상태 가져오기"""
        return self.connection_info