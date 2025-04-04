import mido
import music21
from typing import Dict, List
import numpy as np
from collections import defaultdict

class MIDIFeatureExtractor:
    def __init__(self):
        self.features = {}
    
    def extract_features(self, midi_file: str) -> Dict:
        try:
            score = music21.converter.parse(midi_file)
            flattened_score = score.flatten()
            
            features = {
                'tempo': self._extract_tempo(flattened_score),
                'harmony': self._extract_harmony(flattened_score),
                'rhythm': self._extract_rhythm(flattened_score),
                'melody': self._extract_melody(flattened_score),
                'key_signatures': self._extract_key_signatures(flattened_score),
                'time_signatures': self._extract_time_signatures(flattened_score),
                'instruments': self._extract_instruments(flattened_score)
            }
            
            return features
        except Exception as e:
            print(f"Error extracting features from {midi_file}: {str(e)}")
            return {}
    
    def _extract_tempo(self, score):
        """템포 관련 특징 추출"""
        tempos = []
        for el in score.getElementsByClass('MetronomeMark'):
            tempos.append(el.number)
        
        return {
            'main_tempo': float(np.mean(tempos)) if tempos else None,
            'tempo_changes': len(tempos)
        }
    
    def _extract_harmony(self, score):
        """화성 관련 특징 추출"""
        try:
            chords = score.getElementsByClass('Chord')
            chord_progression = []
            for chord in chords:
                chord_progression.append(str(chord.commonName))
            
            return {
                'chord_progression': chord_progression,
                'unique_chords': len(set(chord_progression))
            }
        except:
            return {'chord_progression': [], 'unique_chords': 0}

    def _extract_rhythm(self, score):
        """리듬 관련 특징 추출"""
        try:
            notes = score.notesAndRests
            durations = [float(note.quarterLength) for note in notes]
            
            return {
                'avg_note_duration': float(np.mean(durations)) if durations else 0,
                'rhythmic_density': len(notes)
            }
        except:
            return {'avg_note_duration': 0, 'rhythmic_density': 0}

    def _extract_melody(self, score):
        """멜로디 관련 특징 추출"""
        try:
            notes = score.getElementsByClass('Note')
            if not notes:
                return {'pitch_range': None, 'avg_pitch': None}
            
            pitches = [note.pitch.midi for note in notes]
            
            return {
                'pitch_range': (min(pitches), max(pitches)) if pitches else None,
                'avg_pitch': float(np.mean(pitches)) if pitches else None
            }
        except:
            return {'pitch_range': None, 'avg_pitch': None}

    def _extract_key_signatures(self, score):
        """조표 추출"""
        try:
            key_signatures = []
            for el in score.getElementsByClass('KeySignature'):
                key_signatures.append(str(el))
            return key_signatures
        except:
            return []

    def _extract_time_signatures(self, score):
        """박자표 추출"""
        try:
            time_signatures = []
            for el in score.getElementsByClass('TimeSignature'):
                time_signatures.append(f"{el.numerator}/{el.denominator}")
            return time_signatures
        except:
            return []

    def _extract_instruments(self, score):
        """사용된 악기 추출"""  
        try:
            instruments = []
            for el in score.getElementsByClass('Instrument'):
                instruments.append(str(el.instrumentName))
            return instruments
        except:
            return []