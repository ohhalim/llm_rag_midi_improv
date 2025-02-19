import mido
import json
from typing import Dict, List, Tuple

class MidiParser:
    def __init__(self, midi_file_path: str):
        self.midi_file = mido.MidiFile(midi_file_path)
        
    def parse_midi(self) -> Dict:
        """MIDI 파일을 파싱하여 트랙, 악기, 노트 정보를 추출합니다."""
        midi_data = {
            'format': self.midi_file.type,
            'ticks_per_beat': self.midi_file.ticks_per_beat,
            'tracks': []
        }
        
        for track_idx, track in enumerate(self.midi_file.tracks):
            track_data = {
                'name': f'Track {track_idx}',
                'events': [],
                'notes': []
            }
            
            current_time = 0
            current_notes = {}  # 현재 연주 중인 노트를 추적
            
            for msg in track:
                current_time += msg.time
                
                if msg.type == 'track_name':
                    track_data['name'] = msg.name
                elif msg.type == 'program_change':
                    track_data['instrument'] = msg.program
                elif msg.type in ['note_on', 'note_off']:
                    event = {
                        'type': msg.type,
                        'note': msg.note,
                        'velocity': msg.velocity,
                        'time': current_time
                    }
                    track_data['events'].append(event)
                    
                    # 노트 온/오프 이벤트를 노트 시퀀스로 변환
                    if msg.type == 'note_on' and msg.velocity > 0:
                        current_notes[msg.note] = current_time
                    elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                        if msg.note in current_notes:
                            start_time = current_notes[msg.note]
                            duration = current_time - start_time
                            note_data = {
                                'note': msg.note,
                                'start_time': start_time,
                                'duration': duration
                            }
                            track_data['notes'].append(note_data)
                            del current_notes[msg.note]
            
            midi_data['tracks'].append(track_data)
        
        return midi_data
    
    def get_notes_as_tuples(self) -> Dict[int, List[Tuple]]:
        """각 트랙의 노트 정보를 music21 형식의 튜플 리스트로 변환합니다."""
        result = {}
        midi_data = self.parse_midi()
        
        for track in midi_data['tracks']:
            if 'instrument' in track:
                notes = []
                for note in track['notes']:
                    # MIDI 노트 번호를 음높이 문자열로 변환
                    pitch = self._midi_note_to_pitch(note['note'])
                    # 틱을 4분음표 기준 duration으로 변환
                    duration = note['duration'] / midi_data['ticks_per_beat']
                    notes.append((pitch, duration))
                result[track['instrument']] = notes
                
        return result
    
    def _midi_note_to_pitch(self, midi_note: int) -> str:
        """MIDI 노트 번호를 음높이 문자열로 변환합니다."""
        notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        octave = (midi_note // 12) - 1
        note = notes[midi_note % 12]
        
        return f"{note}{octave}"

def main():
    # 사용 예시
    parser = MidiParser('moaninfunk.mid')
    
    # MIDI 파일의 전체 구조 파싱
    midi_data = parser.parse_midi()
    print("MIDI 파일 구조:")
    print(json.dumps(midi_data, indent=2))
    
    # music21 형식의 노트 튜플 추출
    note_tuples = parser.get_notes_as_tuples()
    print("\nMusic21 형식의 노트 데이터:")
    for instrument, notes in note_tuples.items():
        print(f"\n악기 번호 {instrument}:")
        print(notes)

if __name__ == "__main__":
    main()