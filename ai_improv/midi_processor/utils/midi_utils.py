"""
MIDI 파일 유틸리티 모듈
MIDI 파일 로드, 저장 및 변환 기능 제공
"""
import io
import json
from mido import MidiFile, MidiTrack, Message, MetaMessage
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_midi_file(file_path):
    """
    MIDI 파일을 로드합니다.
    
    Args:
        file_path: MIDI 파일 경로
        
    Returns:
        MidiFile 객체 또는 None (오류 발생 시)
    """
    try:
        return MidiFile(file_path)
    except Exception as e:
        logger.error(f"MIDI 파일 로드 중 오류: {str(e)}")
        return None

def save_midi_file(mid, output_path):
    """
    MIDI 객체를 파일로 저장합니다.
    
    Args:
        mid: MidiFile 객체
        output_path: 저장할 파일 경로
        
    Returns:
        bool: 성공 여부
    """
    try:
        mid.save(output_path)
        logger.info(f"MIDI 파일이 저장되었습니다: {output_path}")
        return True
    except Exception as e:
        logger.error(f"MIDI 파일 저장 중 오류: {str(e)}")
        return False

def midi_to_bytes(mid):
    """
    MidiFile 객체를 바이트 데이터로 변환합니다.
    
    Args:
        mid: MidiFile 객체
        
    Returns:
        bytes: MIDI 바이트 데이터
    """
    try:
        buffer = io.BytesIO()
        mid.save(file=buffer)
        buffer.seek(0)
        return buffer.read()
    except Exception as e:
        logger.error(f"MIDI를 바이트로 변환 중 오류: {str(e)}")
        return None

def bytes_to_midi(midi_bytes):
    """
    바이트 데이터를 MidiFile 객체로 변환합니다.
    
    Args:
        midi_bytes: MIDI 바이트 데이터
        
    Returns:
        MidiFile: MidiFile 객체
    """
    try:
        buffer = io.BytesIO(midi_bytes)
        return MidiFile(file=buffer)
    except Exception as e:
        logger.error(f"바이트에서 MIDI로 변환 중 오류: {str(e)}")
        return None

def midi_to_json(mid):
    """
    MidiFile 객체를 JSON 형식으로 변환합니다.
    
    Args:
        mid: MidiFile 객체
        
    Returns:
        str: JSON 문자열
    """
    try:
        # JSON 데이터 구조 초기화
        json_data = {
            "tracks": [],
            "time_signatures": [],
            "key_signatures": [],
            "ticks_per_beat": mid.ticks_per_beat
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
                    json_data["time_signatures"].append(f"{msg.numerator}/{msg.denominator}")
                elif msg.type == 'key_signature':
                    json_data["key_signatures"].append(msg.key)
        
        # 초당 틱 계산
        seconds_per_tick = tempo / (1000000 * ticks_per_beat)
        
        # 트랙 처리
        for i, track in enumerate(mid.tracks):
            # 트랙 데이터 초기화
            track_data = {
                "name": track.name if hasattr(track, 'name') else f"Track {i}",
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
            abs_time = 0
            
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
        logger.error(f"MIDI를 JSON으로 변환 중 오류: {str(e)}")
        return json.dumps({"error": str(e)})

def json_to_midi(json_string):
    """
    JSON 문자열을 MidiFile 객체로 변환합니다.
    
    Args:
        json_string: JSON 문자열
        
    Returns:
        MidiFile: MidiFile 객체
    """
    try:
        # JSON 파싱
        json_data = json.loads(json_string)
        
        # MIDI 파일 생성
        mid = MidiFile(ticks_per_beat=json_data.get('ticks_per_beat', 480))
        
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
            
            # 트랙 이름 설정 (있는 경우)
            if "name" in track_data:
                track.append(MetaMessage('track_name', name=track_data["name"], time=0))
            
            # 악기 설정
            instrument = track_data.get('instrument', 0)
            track.append(Message('program_change', program=instrument, time=0))
            
            # 노트 추가
            notes = track_data.get('notes', [])
            notes.sort(key=lambda x: x.get('time', 0))  # 시간 순으로 정렬
            
            ticks_per_beat = json_data.get('ticks_per_beat', 480)
            tempo = 500000  # 기본 템포 (500000 마이크로초/쿼터노트 = 120 BPM)
            
            def seconds_to_ticks(seconds):
                """초 단위 시간을 MIDI 틱으로 변환"""
                return int(seconds * 1000000 * ticks_per_beat / tempo)
            
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
        
        return mid
        
    except Exception as e:
        logger.error(f"JSON을 MIDI로 변환 중 오류: {str(e)}")
        return None 