import mido
import music21
from typing import Dict, List

class MIDIFeatureExtractor:
    def __init__(self):
        self.features = {}
    
    def extract_features(self, midi_file: str) -> Dict:
        """MIDI 파일에서 음악적 특징 추출"""
        score = music21.converter.parse(midi_file)
        
        features = {
            'tempo': self._extract_tempo(score),
            'harmony': self._extract_harmony(score),
            'rhythm': self._extract_rhythm(score),
            'melody': self._extract_melody(score)
        }
        
        return features
    
    def _extract_tempo(self, score):
        # 템포 관련 특징 추출
        pass
    
    def _extract_harmony(self, score):
        # 화성 관련 특징 추출
        pass
    
    def _extract_rhythm(self, score):
        # 리듬 관련 특징 추출
        pass
    
    def _extract_melody(self, score):
        # 멜로디 관련 특징 추출
        pass