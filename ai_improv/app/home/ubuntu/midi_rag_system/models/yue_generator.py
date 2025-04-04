"""
YuE 음악 생성 모듈
YuE 오픈소스 모델을 활용한 음악 생성 기능을 구현합니다.
"""
import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import config
from pathlib import Path
import mido
from mido import MidiFile, MidiTrack, Message, MetaMessage
import io

class YuEMusicGenerator:
    def __init__(self, stage1_model=None, stage2_model=None):
        """
        YuE 음악 생성 모듈 초기화
        
        Args:
            stage1_model: Stage 1 모델 이름 (기본값: config.py에서 설정)
            stage2_model: Stage 2 모델 이름 (기본값: config.py에서 설정)
        """
        self.stage1_model_name = stage1_model or config.YUE_STAGE1_MODEL
        self.stage2_model_name = stage2_model or config.YUE_STAGE2_MODEL
        self.max_new_tokens = config.YUE_MAX_NEW_TOKENS
        self.repetition_penalty = config.YUE_REPETITION_PENALTY
        
        # 모델과 토크나이저는 필요할 때 로드
        self.stage1_model = None
        self.stage1_tokenizer = None
        self.stage2_model = None
        self.stage2_tokenizer = None
    
    def _load_models(self):
        """모델 로드 - M1 최적화 버전"""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        
        # M1 최적화 설정 확인
        use_8bit = getattr(config, 'YUE_LOAD_IN_8BIT', True)
        device_map = "auto"
        
        # 토크나이저 로드
        self.stage1_tokenizer = AutoTokenizer.from_pretrained(config.YUE_STAGE1_MODEL)
        self.stage2_tokenizer = AutoTokenizer.from_pretrained(config.YUE_STAGE2_MODEL)
        
        # 모델 로드 (8비트 양자화 적용)
        print("Stage 1 모델 로드 중...")
        self.stage1_model = AutoModelForCausalLM.from_pretrained(
            config.YUE_STAGE1_MODEL,
            load_in_8bit=use_8bit,
            device_map=device_map,
            torch_dtype=torch.float16
        )
        
        print("Stage 2 모델 로드 중...")
        self.stage2_model = AutoModelForCausalLM.from_pretrained(
            config.YUE_STAGE2_MODEL,
            load_in_8bit=use_8bit,
            device_map=device_map,
            torch_dtype=torch.float16
        )
        
        print("모델 로드 완료")

    
    def generate_music(self, genre_text, lyrics_text, run_n_segments=2, stage2_batch_size=4):
        """
        YuE 모델을 사용하여 음악 생성
        
        Args:
            genre_text: 장르 및 스타일 텍스트
            lyrics_text: 가사 텍스트
            run_n_segments: 생성할 세그먼트 수
            stage2_batch_size: Stage 2 배치 크기
            
        Returns:
            bytes: 생성된 MIDI 파일 바이트
        """
        # 모델 로드
        self._load_models()
        
        # Stage 1: 음악 구조 생성
        print("Stage 1: 음악 구조 생성 중...")
        prompt = f"{genre_text}\n\n{lyrics_text}"
        
        inputs = self.stage1_tokenizer(prompt, return_tensors="pt").to(self.stage1_model.device)
        outputs = self.stage1_model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=True,
            temperature=0.7,
            repetition_penalty=self.repetition_penalty,
            num_return_sequences=1
        )
        
        stage1_output = self.stage1_tokenizer.decode(outputs[0], skip_special_tokens=True)
        print("Stage 1 완료")
        
        # Stage 2: 음악 생성
        print("Stage 2: 음악 생성 중...")
        # 실제 YuE 구현에서는 여기서 Stage 1 출력을 파싱하고 Stage 2 모델로 음악을 생성합니다.
        # 간소화를 위해 여기서는 JSON 형식의 MIDI 데이터를 직접 생성합니다.
        
        # 샘플 JSON 데이터 생성 (실제로는 Stage 2 모델의 출력을 사용)
        midi_data = self._generate_sample_midi_json()
        
        # JSON을 MIDI로 변환
        midi_bytes = self._convert_json_to_midi(midi_data)
        print("Stage 2 완료")
        
        return midi_bytes
    
    def _generate_sample_midi_json(self):
        """
        샘플 MIDI JSON 데이터 생성 (실제 구현에서는 Stage 2 모델 출력 사용)
        
        Returns:
            dict: MIDI JSON 데이터
        """
        # 샘플 MIDI 데이터 생성
        notes = []
        for i in range(100):  # 100개의 노트 생성
            pitch = 60 + (i % 12)  # C4부터 시작하여 한 옥타브 내에서 순환
            time = i * 0.5  # 각 노트는 0.5초 간격
            duration = 0.4  # 노트 지속 시간
            velocity = 64 + (i % 32)  # 세기 변화
            
            notes.append({
                "pitch": pitch,
                "time": time,
                "duration": duration,
                "velocity": velocity
            })
        
        # MIDI JSON 데이터 구성
        midi_data = {
            "tracks": [
                {
                    "instrument": 0,  # 피아노
                    "notes": notes
                }
            ],
            "time_signatures": ["4/4"],
            "key_signatures": []
        }
        
        return midi_data
    
    def _convert_json_to_midi(self, data):
        """
        JSON 데이터를 MIDI 파일로 변환
        
        Args:
            data: MIDI JSON 데이터
            
        Returns:
            bytes: MIDI 파일 바이트
        """
        try:
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
                    delta_ticks = time_ticks - last_time if time_ticks > last_time else 0
                    
                    # 노트 온
                    track.append(Message('note_on', note=pitch, velocity=velocity, time=delta_ticks))
                    
                    # 노트 오프 (노트 온 후에 duration_ticks 시간 후)
                    track.append(Message('note_off', note=pitch, velocity=0, time=duration_ticks))
                    
                    # 마지막 이벤트 시간 업데이트
                    last_time = time_ticks + duration_ticks
            
            # 메모리에 MIDI 파일 생성
            buffer = io.BytesIO()
            mid.save(file=buffer)
            buffer.seek(0)
            return buffer.read()
            
        except Exception as e:
            print(f"MIDI 변환 중 오류 발생: {str(e)}")
            return None
    
    def save_midi(self, midi_data, output_path):
        """
        생성된 MIDI 데이터를 파일로 저장
        
        Args:
            midi_data: MIDI 바이트 데이터
            output_path: 저장할 파일 경로
            
        Returns:
            bool: 저장 성공 여부
        """
        try:
            # 출력 디렉토리 확인
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # MIDI 바이너리 데이터 저장
            with open(output_path, 'wb') as f:
                f.write(midi_data)
            
            print(f"MIDI 파일이 저장되었습니다: {output_path}")
            return True
        except Exception as e:
            print(f"MIDI 파일 저장 중 오류 발생: {str(e)}")
            return False
