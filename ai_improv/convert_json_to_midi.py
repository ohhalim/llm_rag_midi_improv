import json
import mido
from mido import Message, MidiFile, MidiTrack

def convert_json_to_midi(json_file, midi_file):
    """JSON 파일을 표준 MIDI 파일로 변환합니다."""
    # JSON 파일 로드
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # MIDI 파일 생성
    mid = MidiFile()
    
    # 템포 트랙 추가
    tempo_track = MidiTrack()
    mid.tracks.append(tempo_track)
    
    # 템포 설정 (120 BPM)
    tempo_track.append(mido.MetaMessage('set_tempo', tempo=500000, time=0))
    
    # 타임 시그니처 설정
    time_signatures = data.get('time_signatures', ['4/4'])
    if time_signatures and len(time_signatures) > 0:
        time_sig = time_signatures[0].split('/')
        if len(time_sig) == 2:
            numerator, denominator = int(time_sig[0]), int(time_sig[1])
            tempo_track.append(mido.MetaMessage('time_signature', 
                                                numerator=numerator, 
                                                denominator=denominator,
                                                clocks_per_click=24,
                                                notated_32nd_notes_per_beat=8,
                                                time=0))
    
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
    
    # MIDI 파일 저장
    mid.save(midi_file)
    print(f"MIDI 파일이 생성되었습니다: {midi_file}")

# 실행 예시
json_file = "/Users/ohhalim/git_box/llm_rag_midi_improv/ai_improv/data/output/output.mid"  # 실제로는 JSON 파일
midi_file = "/Users/ohhalim/git_box/llm_rag_midi_improv/ai_improv/data/output/converted_output.mid"  # 변환될 MIDI 파일
convert_json_to_midi(json_file, midi_file)