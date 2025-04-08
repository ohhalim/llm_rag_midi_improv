"""
YuE 음악 생성 모듈 - MIDI 솔로라인 특화 버전
YuE 오픈소스 모델을 활용한 솔로라인 생성 기능을 구현합니다.
"""
import os
import json
import tempfile
import torch
from pathlib import Path
import logging
import io
import mido
from mido import MidiFile, MidiTrack, Message, MetaMessage
import numpy as np
import config

# 로깅 설정
logging.basicConfig(level=getattr(logging, config.LOG_LEVEL, "INFO"), 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("YuEGenerator")

class YuESoloGenerator:
    def __init__(self):
        """YuE 기반 솔로라인 생성기 초기화"""
        # 모델 경로 설정
        self.stage1_model_name = config.YUE_STAGE1_MODEL
        self.stage2_model_name = config.YUE_STAGE2_MODEL
        
        # 생성 매개변수
        self.max_new_tokens = config.YUE_MAX_NEW_TOKENS
        self.repetition_penalty = config.YUE_REPETITION_PENALTY
        
        # 메모리 최적화 설정
        self.run_n_segments = getattr(config, 'YUE_RUN_N_SEGMENTS', 1)
        self.stage2_batch_size = getattr(config, 'YUE_STAGE2_BATCH_SIZE', 1)
        self.use_flash_attention = getattr(config, 'YUE_USE_FLASH_ATTENTION', False)
        
        # 모델과 토크나이저는 필요할 때 로드
        self.stage1_model = None
        self.stage1_tokenizer = None
        self.stage2_model = None
        self.stage2_tokenizer = None
        
        logger.info(f"YuE 솔로라인 생성기 초기화 완료 (모델: {self.stage1_model_name}, {self.stage2_model_name})")
    
    def _load_models(self):
        """필요시 YuE 모델 로드 - 메모리 최적화 버전"""
        try:
            logger.info("YuE 모델 로드 중...")
            
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
            
            # 8비트 양자화 설정
            load_in_8bit = getattr(config, 'YUE_LOAD_IN_8BIT', True)
            device_map = "auto"
            
            # 스테이지 1 (솔로라인 계획 생성) 모델 로드
            if self.stage1_tokenizer is None:
                logger.info(f"스테이지 1 토크나이저 로드 중: {self.stage1_model_name}")
                self.stage1_tokenizer = AutoTokenizer.from_pretrained(self.stage1_model_name)
            
            if self.stage1_model is None:
                logger.info(f"스테이지 1 모델 로드 중: {self.stage1_model_name}")
                self.stage1_model = AutoModelForCausalLM.from_pretrained(
                    self.stage1_model_name,
                    load_in_8bit=load_in_8bit,
                    device_map=device_map,
                    torch_dtype=torch.float16
                )
            
            # 스테이지 2 (구체적인 MIDI 노트 생성) 모델 로드
            if self.stage2_tokenizer is None:
                logger.info(f"스테이지 2 토크나이저 로드 중: {self.stage2_model_name}")
                self.stage2_tokenizer = AutoTokenizer.from_pretrained(self.stage2_model_name)
            
            if self.stage2_model is None:
                logger.info(f"스테이지 2 모델 로드 중: {self.stage2_model_name}")
                self.stage2_model = AutoModelForCausalLM.from_pretrained(
                    self.stage2_model_name,
                    load_in_8bit=load_in_8bit,
                    device_map=device_map,
                    torch_dtype=torch.float16
                )
            
            logger.info("YuE 모델 로드 완료")
            return True
            
        except Exception as e:
            logger.error(f"모델 로드 중 오류 발생: {str(e)}")
            return False
    
    def generate_solo(self, input_midi, reference_midi=None, genre_text=None, lyrics_text=None, solo_instrument=0):
        """
        YuE 모델을 사용하여 입력 MIDI에 어울리는 솔로라인 생성
        
        Args:
            input_midi: 입력 MIDI 파일 경로
            reference_midi: 참조할 MIDI 파일 경로 (선택적)
            genre_text: 장르 및 스타일 텍스트 (선택적)
            lyrics_text: 가사 텍스트 (선택적)
            solo_instrument: 솔로 악기 번호 (기본값: 0, 피아노)
            
        Returns:
            bytes: 생성된 MIDI 파일 바이트
        """
        try:
            # 기본 장르 및 가사 설정
            if not genre_text:
                genre_text = "inspiring instrumental uplifting solo improvisation"
            
            if not lyrics_text:
                lyrics_text = """[verse]
This is a solo improvisation

[chorus]
Pure instrumental expression"""
            
            # YuE 모델을 사용하여 솔로라인 생성
            temp_dir = tempfile.mkdtemp()
            
            # genre.txt 생성
            genre_path = os.path.join(temp_dir, "genre.txt")
            with open(genre_path, "w", encoding="utf-8") as f:
                f.write(genre_text)
            
            # lyrics.txt 생성
            lyrics_path = os.path.join(temp_dir, "lyrics.txt")
            with open(lyrics_path, "w", encoding="utf-8") as f:
                f.write(lyrics_text)
            
            # 출력 디렉토리 설정
            output_dir = os.path.join(temp_dir, "output")
            os.makedirs(output_dir, exist_ok=True)
            
            # YuE 스크립트 실행
            cmd = self._build_yue_command(
                genre_path=genre_path,
                lyrics_path=lyrics_path,
                output_dir=output_dir,
                reference_midi=reference_midi
            )
            
            logger.info(f"YuE 명령 실행: {cmd}")
            os.system(cmd)
            
            # 생성된 MIDI 파일 찾기
            generated_midi = None
            for file in os.listdir(output_dir):
                if file.endswith(".mid") or file.endswith(".midi"):
                    generated_midi = os.path.join(output_dir, file)
                    break
            
            if not generated_midi:
                logger.warning("YuE에서 MIDI 파일을 생성하지 못했습니다. 대체 솔로 생성")
                return self._generate_fallback_solo(solo_instrument)
            
            # 생성된 MIDI 파일 로드 및 솔로라인 추출
            logger.info(f"생성된 MIDI 파일: {generated_midi}")
            solo_midi = self._convert_yue_to_solo(generated_midi, solo_instrument)
            
            # 임시 디렉토리 정리
            import shutil
            shutil.rmtree(temp_dir)
            
            return solo_midi
            
        except Exception as e:
            logger.error(f"YuE 솔로라인 생성 중 오류 발생: {str(e)}")
            return self._generate_fallback_solo(solo_instrument)
    
    def _build_yue_command(self, genre_path, lyrics_path, output_dir, reference_midi=None):
        """YuE 실행 명령 구성"""
        # YuE 디렉토리 확인 (config에서 설정되어 있어야 함)
        yue_dir = getattr(config, 'YUE_DIR', os.path.expanduser("~/YuE/inference"))
        
        # 기본 명령
        cmd = f"cd {yue_dir} && python infer.py "
        cmd += f"--cuda_idx 0 "
        cmd += f"--stage1_model {self.stage1_model_name} "
        cmd += f"--stage2_model {self.stage2_model_name} "
        cmd += f"--genre_txt {genre_path} "
        cmd += f"--lyrics_txt {lyrics_path} "
        cmd += f"--run_n_segments {self.run_n_segments} "
        cmd += f"--stage2_batch_size {self.stage2_batch_size} "
        cmd += f"--output_dir {output_dir} "
        cmd += f"--max_new_tokens {self.max_new_tokens} "
        cmd += f"--repetition_penalty {self.repetition_penalty} "
        
        # 참조 MIDI가 있는 경우 ICL 모드 활성화
        if reference_midi:
            # ICL 모델로 변경
            cmd = cmd.replace("YuE-s1-7B-anneal-en-cot", "YuE-s1-7B-anneal-en-icl")
            
            # ICL 관련 옵션 추가
            cmd += f"--use_audio_prompt "
            cmd += f"--audio_prompt_path {reference_midi} "
            cmd += f"--prompt_start_time 0 "
            cmd += f"--prompt_end_time 30 "
        
        return cmd
    
    def _convert_yue_to_solo(self, yue_midi_path, solo_instrument=0):
        """YuE가 생성한 MIDI에서 솔로라인만 추출"""
        try:
            # MIDI 파일 로드
            mid = MidiFile(yue_midi_path)
            
            # 솔로 트랙을 위한 새 MIDI 파일 생성
            solo_mid = MidiFile(ticks_per_beat=mid.ticks_per_beat)
            
            # 템포 및 메타 트랙 복사
            if len(mid.tracks) > 0:
                meta_track = MidiTrack()
                solo_mid.tracks.append(meta_track)
                
                # 메타 이벤트만 복사
                for msg in mid.tracks[0]:
                    if not isinstance(msg, Message) or msg.type in ['time_signature', 'key_signature', 'set_tempo']:
                        meta_track.append(msg)
            
            # 솔로 트랙 생성
            solo_track = MidiTrack()
            solo_mid.tracks.append(solo_track)
            
            # 악기 설정
            solo_track.append(Message('program_change', program=solo_instrument, time=0))
            
            # YuE 출력에서 멜로디 트랙 찾기 (일반적으로 보컬 트랙)
            melody_track = None
            for track in mid.tracks[1:]:  # 첫 번째 트랙은 메타 트랙이므로 건너뜀
                # 트랙의 프로그램 변경 메시지 확인
                program = 0
                for msg in track:
                    if msg.type == 'program_change':
                        program = msg.program
                        break
                
                # 멜로디 트랙 선택 (보컬이나 리드 악기 우선)
                if program in [0, 24, 25, 65, 73, 80]:  # 피아노, 기타, 색소폰 등 솔로 악기
                    melody_track = track
                    break
            
            # 멜로디 트랙을 찾지 못한 경우 첫 번째 비메타 트랙 사용
            if melody_track is None and len(mid.tracks) > 1:
                melody_track = mid.tracks[1]
            
            # 멜로디 트랙의 노트 이벤트 복사
            if melody_track:
                for msg in melody_track:
                    if isinstance(msg, Message) and msg.type in ['note_on', 'note_off']:
                        # 선택한 악기 음색으로 노트 변경
                        solo_track.append(msg)
            
            # MIDI 바이트로 변환
            buffer = io.BytesIO()
            solo_mid.save(file=buffer)
            buffer.seek(0)
            return buffer.read()
            
        except Exception as e:
            logger.error(f"YuE MIDI 변환 중 오류: {str(e)}")
            return self._generate_fallback_solo(solo_instrument)
    
    def _generate_fallback_solo(self, instrument=0):
        """YuE 실패 시 간단한 솔로라인 생성"""
        logger.info(f"대체 솔로라인 생성 중 (악기: {instrument})")
        
        # MIDI 파일 생성
        mid = MidiFile()
        
        # 메타 트랙 추가
        meta_track = MidiTrack()
        mid.tracks.append(meta_track)
        
        # 템포 설정 (120 BPM)
        meta_track.append(MetaMessage('set_tempo', tempo=500000, time=0))
        
        # 타임 시그니처 설정 (4/4)
        meta_track.append(MetaMessage('time_signature', numerator=4, denominator=4, 
                                      clocks_per_click=24, notated_32nd_notes_per_beat=8, time=0))
        
        # 솔로 트랙 추가
        solo_track = MidiTrack()
        mid.tracks.append(solo_track)
        
        # 악기 설정
        solo_track.append(Message('program_change', program=instrument, time=0))
        
        # 주요 음계 정의 (C major)
        c_major_scale = [60, 62, 64, 65, 67, 69, 71, 72]  # C4 ~ C5 (C 메이저 스케일)
        
        # 다양한 리듬 패턴 정의 (각 숫자는 4분음표 기준 길이)
        rhythm_patterns = [
            [1, 1, 1, 1],  # 4개의 4분음표
            [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],  # 8개의 8분음표
            [1.5, 0.5, 1, 1],  # 혼합 패턴 1
            [0.5, 1, 0.5, 1, 1],  # 혼합 패턴 2
            [2, 1, 1],  # 혼합 패턴 3
            [1, 1, 2],  # 혼합 패턴 4
        ]
        
        # 멜로디 패턴 정의 (음계 인덱스)
        melody_patterns = [
            [0, 2, 4, 2],  # 단순 상승 하강
            [0, 4, 7, 4],  # 도미솔미
            [0, 1, 2, 4, 7],  # 순차 상승
            [7, 4, 2, 1, 0],  # 순차 하강
            [0, 3, 2, 0],  # 종적 느낌
            [0, 7, 5, 4],  # 도솔파미
        ]
        
        # 솔로 생성
        time = 0  # 현재 시간 (틱)
        ticks_per_beat = 480  # 4분음표당 틱 수
        
        # 약 30초 분량 생성 (120 BPM에서 약 60 마디)
        for _ in range(20):  # 약 20개의 프레이즈 생성
            # 랜덤하게 음높이 오프셋 선택 (다양한 옥타브)
            octave_offset = np.random.choice([-12, 0, 12])
            
            # 랜덤하게 리듬 패턴 선택
            rhythm = np.random.choice(rhythm_patterns)
            
            # 랜덤하게 멜로디 패턴 선택
            if len(rhythm) <= len(melody_patterns):
                melody_pattern = np.random.choice(melody_patterns)
                melody = melody_pattern[:len(rhythm)]  # 리듬에 맞게 멜로디 자르기
            else:
                # 리듬이 더 길면 멜로디 패턴 반복 또는 확장
                melody_pattern = np.random.choice(melody_patterns)
                melody = []
                while len(melody) < len(rhythm):
                    melody.extend(melody_pattern)
                melody = melody[:len(rhythm)]  # 리듬에 맞게 자르기
            
            # 프레이즈 생성
            for i, (r, m) in enumerate(zip(rhythm, melody)):
                note_ticks = int(r * ticks_per_beat)
                
                # 음표 인덱스가 범위를 벗어나면 조정
                m_idx = m % len(c_major_scale)
                
                # 실제 MIDI 음높이 계산
                pitch = c_major_scale[m_idx] + octave_offset
                
                # 음높이 범위 확인 (33-85)
                pitch = max(33, min(85, pitch))
                
                # 세기 계산 (프레이즈 시작은 강하게, 끝은 약하게)
                if i == 0:
                    velocity = np.random.randint(80, 110)  # 프레이즈 시작
                elif i == len(rhythm) - 1:
                    velocity = np.random.randint(50, 80)  # 프레이즈 끝
                else:
                    velocity = np.random.randint(60, 90)  # 프레이즈 중간
                
                # 노트 온
                solo_track.append(Message('note_on', note=pitch, velocity=velocity, time=0 if i > 0 else time))
                
                # 노트 오프 (95% 길이로 약간의 간격 유지)
                solo_track.append(Message('note_off', note=pitch, velocity=0, time=int(note_ticks * 0.95)))
                
                # 다음 음표 시간 계산
                time = int(note_ticks * 0.05)  # 5%는 다음 note_on까지의 간격
            
            # 프레이즈 간 휴식
            rest_ticks = int(np.random.choice([0.5, 1]) * ticks_per_beat)
            time = rest_ticks
        
        # MIDI 바이트로 변환
        buffer = io.BytesIO()
        mid.save(file=buffer)
        buffer.seek(0)
        return buffer.read()
    
    def to_json(self, midi_data):
        """MIDI 바이트를 JSON 형식으로 변환"""
        try:
            # MIDI 바이트를 파일 객체로 변환
            buffer = io.BytesIO(midi_data)
            mid = MidiFile(file=buffer)
            
            # JSON 데이터 구조 초기화
            json_data = {
                "tracks": [],
                "time_signatures": ["4/4"],  # 기본값
                "key_signatures": []
            }
            
            # 틱당 시간(초) 계산
            tempo = 500000  # 기본 템포 (500000 마이크로초/쿼터노트 = 120 BPM)
            ticks_per_beat = mid.ticks_per_beat
            
            # 템포 및 시간/키 시그니처 찾기
            for track in mid.tracks:
                for msg in track:
                    if msg.type == 'set_tempo':
                        tempo = msg.tempo
                    elif msg.type == 'time_signature':
                        json_data["time_signatures"] = [f"{msg.numerator}/{msg.denominator}"]
                    elif msg.type == 'key_signature':
                        json_data["key_signatures"].append(msg.key)
            
            # 초당 틱 계산
            seconds_per_tick = tempo / (1000000 * ticks_per_beat)
            
            # 트랙 처리
            for i, track in enumerate(mid.tracks):
                # 메타 트랙 건너뛰기 (첫 번째 트랙은 일반적으로 메타 트랙)
                if i == 0 and all(not isinstance(msg, Message) or msg.type not in ['note_on', 'note_off'] for msg in track):
                    continue
                
                # 트랙 데이터 초기화
                track_data = {
                    "instrument": 0,  # 기본 악기 (피아노)
                    "notes": []
                }
                
                # 악기 찾기
                for msg in track:
                    if msg.type == 'program_change':
                        track_data["instrument"] = msg.program
                        break
                
                # 노트 이벤트 처리
                notes_on = {}  # 활성 노트 추적
                abs_time = 0  # 절대 시간 (틱)
                
                for msg in track:
                    # 델타 시간 추가
                    abs_time += msg.time
                    
                    if msg.type == 'note_on' and msg.velocity > 0:
                        # 노트 시작
                        notes_on[msg.note] = {
                            "start_time": abs_time,
                            "velocity": msg.velocity
                        }
                    elif (msg.type == 'note_off') or (msg.type == 'note_on' and msg.velocity == 0):
                        # 노트 종료
                        if msg.note in notes_on:
                            start_time = notes_on[msg.note]["start_time"]
                            duration_ticks = abs_time - start_time
                            
                            # 틱을 초로 변환
                            start_seconds = start_time * seconds_per_tick
                            duration_seconds = duration_ticks * seconds_per_tick
                            
                            # 노트 추가
                            track_data["notes"].append({
                                "pitch": msg.note,
                                "time": start_seconds,
                                "duration": duration_seconds,
                                "velocity": notes_on[msg.note]["velocity"]
                            })
                            
                            # 활성 노트에서 제거
                            del notes_on[msg.note]
                
                # 트랙에 노트가 있으면 추가
                if track_data["notes"]:
                    json_data["tracks"].append(track_data)
            
            return json.dumps(json_data, indent=2)
            
        except Exception as e:
            logger.error(f"MIDI를 JSON으로 변환 중 오류 발생: {str(e)}")
            return json.dumps({"error": str(e)})
    
    def from_json(self, json_string):
        """JSON 형식의 MIDI 데이터를 MIDI 바이트로 변환"""
        try:
            # JSON 파싱
            json_data = json.loads(json_string)
            
            # MIDI 파일 생성
            mid = MidiFile()
            
            # 템포 트랙 추가
            tempo_track = MidiTrack()
            mid.tracks.append(tempo_track)
            
            # 템포 설정 (120 BPM)
            tempo_track.append(MetaMessage('set_tempo', tempo=500000, time=0))
            
            # 타임 시그니처 설정
            time_signatures = json_data.get('time_signatures', ['4/4'])
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
            key_signatures = json_data.get('key_signatures', [])
            if key_signatures and len(key_signatures) > 0:
                tempo_track.append(MetaMessage('key_signature', key=key_signatures[0], time=0))
            
            # 트랙 추가
            for track_data in json_data.get('tracks', []):
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
            logger.error(f"JSON을 MIDI로 변환 중 오류 발생: {str(e)}")
            return None
    
    def save_midi(self, midi_data, output_path):
        """생성된 MIDI 데이터를 파일로 저장"""
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