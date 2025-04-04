"""
MIDI 특징 추출기 테스트 모듈
MIDIFeatureExtractor 클래스의 기능을 테스트합니다.
"""
import unittest
import os
import sys
import tempfile
from pathlib import Path

# 프로젝트 루트 디렉토리를 Python 경로에 추가
sys.path.append(str(Path(__file__).resolve().parent.parent))

from midi_rag_system.core.midi_feature_extractor import MIDIFeatureExtractor

class TestMIDIFeatureExtractor(unittest.TestCase):
    """MIDIFeatureExtractor 테스트 클래스"""
    
    def setUp(self):
        """테스트 설정"""
        self.extractor = MIDIFeatureExtractor()
        
        # 테스트용 MIDI 파일 생성
        self.test_midi_path = self._create_test_midi()
    
    def tearDown(self):
        """테스트 정리"""
        # 테스트용 MIDI 파일 삭제
        if os.path.exists(self.test_midi_path):
            os.remove(self.test_midi_path)
    
    def _create_test_midi(self):
        """테스트용 MIDI 파일 생성"""
        import mido
        from mido import MidiFile, MidiTrack, Message, MetaMessage
        
        # 임시 파일 생성
        fd, path = tempfile.mkstemp(suffix='.mid')
        os.close(fd)
        
        # MIDI 파일 생성
        mid = MidiFile()
        track = MidiTrack()
        mid.tracks.append(track)
        
        # 메타 메시지 추가
        track.append(MetaMessage('set_tempo', tempo=500000, time=0))
        track.append(MetaMessage('time_signature', numerator=4, denominator=4, time=0))
        
        # 노트 추가
        for i in range(10):
            note = 60 + i % 12  # C4부터 시작하여 한 옥타브 내에서 순환
            track.append(Message('note_on', note=note, velocity=64, time=0))
            track.append(Message('note_off', note=note, velocity=0, time=480))
        
        # MIDI 파일 저장
        mid.save(path)
        
        return path
    
    def test_extract_features(self):
        """특징 추출 테스트"""
        features = self.extractor.extract_features(self.test_midi_path)
        
        # 특징이 추출되었는지 확인
        self.assertIsNotNone(features)
        self.assertIsInstance(features, dict)
        
        # 필수 특징이 포함되어 있는지 확인
        self.assertIn('tempo', features)
        self.assertIn('harmony', features)
        self.assertIn('rhythm', features)
        self.assertIn('melody', features)
        self.assertIn('key_signatures', features)
        self.assertIn('time_signatures', features)
        self.assertIn('instruments', features)
    
    def test_extract_tempo(self):
        """템포 추출 테스트"""
        features = self.extractor.extract_features(self.test_midi_path)
        
        # 템포 특징이 올바른 형식인지 확인
        self.assertIn('tempo', features)
        self.assertIn('main_tempo', features['tempo'])
        self.assertIn('tempo_changes', features['tempo'])
    
    def test_extract_melody(self):
        """멜로디 추출 테스트"""
        features = self.extractor.extract_features(self.test_midi_path)
        
        # 멜로디 특징이 올바른 형식인지 확인
        self.assertIn('melody', features)
        self.assertIn('pitch_range', features['melody'])
        self.assertIn('avg_pitch', features['melody'])
    
    def test_nonexistent_file(self):
        """존재하지 않는 파일 테스트"""
        features = self.extractor.extract_features("nonexistent_file.mid")
        
        # 오류 발생 시 빈 딕셔너리 반환 확인
        self.assertEqual(features, {})

if __name__ == '__main__':
    unittest.main()
