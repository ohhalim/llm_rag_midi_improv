from music21 import *
import mido
import numpy as np
from collections import defaultdict
from typing import List, Dict, Tuple

class AdvancedMIDIFeatureExtractor:
    def __init__(self):
        self.features = {}
        
    def extract_harmony_features(self, midi_file: str) -> Dict:
        """화성 진행 특징 추출"""
        score = converter.parse(midi_file)
        
        harmony_features = {
            'chord_progressions': [],
            'key_changes': [],
            'chord_density': 0,
            'common_progressions': defaultdict(int),
            'harmony_complexity': 0
        }
        
        # 화음 추출
        chords = []
        for chord in score.chordify().recurse().getElementsByClass('Chord'):
            # 화음 정보 저장
            chord_name = chord.commonName
            chords.append(chord_name)
            
            # 연속된 두 화음의 진행 패턴 카운트
            if len(chords) >= 2:
                progression = f"{chords[-2]}->{chords[-1]}"
                harmony_features['common_progressions'][progression] += 1
        
        harmony_features['chord_progressions'] = chords
        
        # 조성 변화 탐지
        prev_key = None
        for key_sig in score.recurse().getElementsByClass('KeySignature'):
            current_key = key_sig.asKey()
            if prev_key and current_key != prev_key:
                harmony_features['key_changes'].append(str(current_key))
            prev_key = current_key
            
        # 화성 복잡도 계산 (사용된 고유 화음 수 기준)
        unique_chords = len(set(chords))
        harmony_features['harmony_complexity'] = unique_chords
        
        # 마디당 평균 화음 수
        total_measures = len(score.measureRange(0, len(score.measures())))
        harmony_features['chord_density'] = len(chords) / total_measures if total_measures > 0 else 0
        
        return harmony_features
    
    def extract_rhythm_features(self, midi_file: str) -> Dict:
        """리듬 패턴 특징 추출"""
        mid = mido.MidiFile(midi_file)
        
        rhythm_features = {
            'rhythm_patterns': defaultdict(int),
            'tempo_changes': [],
            'time_signature_changes': [],
            'syncopation_score': 0,
            'rhythmic_density': defaultdict(float),
            'common_durations': defaultdict(int)
        }
        
        for track in mid.tracks:
            current_time = 0
            current_measure = 0
            notes_in_measure = []
            
            for msg in track:
                current_time += msg.time
                
                if msg.type == 'note_on' and msg.velocity > 0:
                    # 노트의 상대적 위치 계산 (마디 내)
                    position_in_measure = current_time % 480  # 가정: 480 ticks = 1마디
                    notes_in_measure.append(position_in_measure)
                    
                    # 노트 길이 패턴 저장
                    if hasattr(msg, 'time'):
                        duration_key = f"1/{int(480/msg.time)}" if msg.time > 0 else "1/32"
                        rhythm_features['common_durations'][duration_key] += 1
                
                elif msg.type == 'time_signature':
                    rhythm_features['time_signature_changes'].append(
                        f"{msg.numerator}/{msg.denominator}"
                    )
                
                # 마디가 바뀔 때마다 리듬 패턴 분석
                if current_time // 480 > current_measure:
                    if notes_in_measure:
                        # 리듬 패턴을 문자열로 변환하여 저장
                        pattern = '-'.join(str(int(pos/30)) for pos in sorted(notes_in_measure))
                        rhythm_features['rhythm_patterns'][pattern] += 1
                        
                        # 리듬 밀도 계산 (마디당 노트 수)
                        rhythm_features['rhythmic_density'][current_measure] = len(notes_in_measure)
                        
                        # 당김음 점수 계산
                        syncopation = self._calculate_syncopation(notes_in_measure)
                        rhythm_features['syncopation_score'] += syncopation
                    
                    current_measure += 1
                    notes_in_measure = []
        
        # 전체 당김음 점수 정규화
        if current_measure > 0:
            rhythm_features['syncopation_score'] /= current_measure
        
        return rhythm_features
    
    def _calculate_syncopation(self, note_positions: List[float]) -> float:
        """당김음 점수 계산"""
        # 기본 비트 위치 (4분의 4박자 기준)
        strong_beats = [0, 120, 240, 360]  # 강박 위치
        score = 0
        
        for pos in note_positions:
            # 가장 가까운 강박과의 거리 계산
            min_distance = min(abs(pos - beat) for beat in strong_beats)
            # 강박에서 멀수록 당김음 점수 증가
            score += min_distance / 120  # 120 ticks = 1박
        
        return score / len(note_positions) if note_positions else 0
    
    def extract_melodic_features(self, midi_file: str) -> Dict:
        """멜로디 특징 추출"""
        score = converter.parse(midi_file)
        
        melodic_features = {
            'contour': [],
            'intervals': defaultdict(int),
            'phrase_lengths': [],
            'melodic_range': {'min': float('inf'), 'max': float('-inf')},
            'common_motifs': defaultdict(int)
        }
        
        # 주요 멜로디 라인 찾기
        melody_part = None
        max_notes = 0
        
        for part in score.parts:
            note_count = len(part.flatten().notes)
            if note_count > max_notes:
                max_notes = note_count
                melody_part = part
        
        if melody_part:
            notes = melody_part.flatten().notes
            prev_note = None
            current_phrase = []
            
            for note in notes:
                if isinstance(note, note.Note):
                    # 음높이 범위 업데이트
                    pitch = note.pitch.midi
                    melodic_features['melodic_range']['min'] = min(
                        melodic_features['melodic_range']['min'], pitch)
                    melodic_features['melodic_range']['max'] = max(
                        melodic_features['melodic_range']['max'], pitch)
                    
                    # 멜로디 윤곽 저장
                    melodic_features['contour'].append(pitch)
                    
                    # 음정 간격 분석
                    if prev_note and isinstance(prev_note, note.Note):
                        interval = abs(pitch - prev_note.pitch.midi)
                        interval_name = interval % 12  # 옥타브 정규화
                        melodic_features['intervals'][interval_name] += 1
                    
                    # 프레이즈 길이 분석
                    if note.duration.quarterLength > 1.5:  # 긴 음표를 프레이즈 경계로 가정
                        if current_phrase:
                            melodic_features['phrase_lengths'].append(len(current_phrase))
                            # 모티프 추출 (3-4개 음표 시퀀스)
                            if len(current_phrase) >= 3:
                                for i in range(len(current_phrase)-2):
                                    motif = tuple(current_phrase[i:i+3])
                                    melodic_features['common_motifs'][motif] += 1
                        current_phrase = []
                    else:
                        current_phrase.append(pitch)
                    
                    prev_note = note
        
        return melodic_features
    
    def extract_all_features(self, midi_file: str) -> Dict:
        """모든 음악적 특징 추출"""
        features = {
            'harmony': self.extract_harmony_features(midi_file),
            'rhythm': self.extract_rhythm_features(midi_file),
            'melody': self.extract_melodic_features(midi_file)
        }
        
        return features

def format_features_for_llm(features: Dict) -> str:
    """특징을 LLM이 이해할 수 있는 텍스트로 변환"""
    text = "Musical Analysis:\n\n"
    
    # Harmony Features
    text += "Harmony:\n"
    text += f"- Harmony complexity: {features['harmony']['harmony_complexity']}\n"
    text += "- Common chord progressions:\n"
    for prog, count in list(features['harmony']['common_progressions'].items())[:5]:
        text += f"  * {prog}: {count} times\n"
    text += f"- Key changes: {', '.join(features['harmony']['key_changes'])}\n"
    
    # Rhythm Features
    text += "\nRhythm:\n"
    text += f"- Syncopation score: {features['rhythm']['syncopation_score']:.2f}\n"
    text += "- Common note durations:\n"
    for duration, count in list(features['rhythm']['common_durations'].items())[:5]:
        text += f"  * {duration}: {count} times\n"
    text += f"- Time signatures: {', '.join(features['rhythm']['time_signature_changes'])}\n"
    
    # Melody Features
    text += "\nMelody:\n"
    text += f"- Pitch range: {features['melody']['melodic_range']['min']} to {features['melody']['melodic_range']['max']}\n"
    text += "- Common intervals:\n"
    for interval, count in list(features['melody']['intervals'].items())[:5]:
        text += f"  * {interval} semitones: {count} times\n"
    if features['melody']['phrase_lengths']:
        avg_phrase = sum(features['melody']['phrase_lengths']) / len(features['melody']['phrase_lengths'])
        text += f"- Average phrase length: {avg_phrase:.1f} notes\n"
    
    return text

# 사용 예시
def main():
    extractor = AdvancedMIDIFeatureExtractor()
    midi_file = "example.mid"
    
    # 모든 특징 추출
    features = extractor.extract_all_features(midi_file)
    
    # LLM용 텍스트로 변환
    text_features = format_features_for_llm(features)
    print(text_features)

if __name__ == "__main__":
    main()