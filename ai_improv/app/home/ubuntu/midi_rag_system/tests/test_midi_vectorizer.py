"""
MIDI 벡터화 모듈 테스트
MIDIVectorizer 클래스의 기능을 테스트합니다.
"""
import unittest
import os
import sys
import tempfile
from pathlib import Path

# 프로젝트 루트 디렉토리를 Python 경로에 추가
sys.path.append(str(Path(__file__).resolve().parent.parent))

from midi_rag_system.core.midi_vectorizer import MIDIVectorizer
from midi_rag_system.core.midi_feature_extractor import MIDIFeatureExtractor

class TestMIDIVectorizer(unittest.TestCase):
    """MIDIVectorizer 테스트 클래스"""
    
    def setUp(self):
        """테스트 설정"""
        # 테스트용 임베딩 모델 설정 (실제 Ollama 서버 없이 테스트)
        self.vectorizer = self._create_test_vectorizer()
        
        # 테스트용 MIDI 파일 생성
        self.test_midi_paths = self._create_test_midi_files(3)
        
        # 테스트용 벡터 저장소 경로
        self.test_vectorstore_path = tempfile.mkdtemp()
    
    def tearDown(self):
        """테스트 정리"""
        # 테스트용 MIDI 파일 삭제
        for path in self.test_midi_paths:
            if os.path.exists(path):
                os.remove(path)
        
        # 테스트용 벡터 저장소 삭제
        if os.path.exists(self.test_vectorstore_path):
            import shutil
            shutil.rmtree(self.test_vectorstore_path)
    
    def _create_test_vectorizer(self):
        """테스트용 벡터화 모듈 생성 (모의 임베딩 사용)"""
        from unittest.mock import MagicMock
        
        # 실제 MIDIVectorizer 인스턴스 생성
        vectorizer = MIDIVectorizer()
        
        # 임베딩 모델을 모의 객체로 대체
        mock_embeddings = MagicMock()
        mock_embeddings.embed_documents.return_value = [[0.1, 0.2, 0.3, 0.4] for _ in range(10)]
        mock_embeddings.embed_query.return_value = [0.1, 0.2, 0.3, 0.4]
        
        vectorizer.embeddings = mock_embeddings
        
        return vectorizer
    
    def _create_test_midi_files(self, count):
        """테스트용 MIDI 파일 여러 개 생성"""
        import mido
        from mido import MidiFile, MidiTrack, Message, MetaMessage
        
        paths = []
        for i in range(count):
            # 임시 파일 생성
            fd, path = tempfile.mkstemp(suffix=f'_test{i}.mid')
            os.close(fd)
            
            # MIDI 파일 생성
            mid = MidiFile()
            track = MidiTrack()
            mid.tracks.append(track)
            
            # 메타 메시지 추가
            track.append(MetaMessage('set_tempo', tempo=500000, time=0))
            track.append(MetaMessage('time_signature', numerator=4, denominator=4, time=0))
            
            # 노트 추가 (각 파일마다 다른 노트 패턴)
            for j in range(10):
                note = 60 + (i * 5 + j) % 12  # 파일마다 다른 시작 노트
                track.append(Message('note_on', note=note, velocity=64, time=0))
                track.append(Message('note_off', note=note, velocity=0, time=480))
            
            # MIDI 파일 저장
            mid.save(path)
            paths.append(path)
        
        return paths
    
    def test_vectorize_midi(self):
        """MIDI 벡터화 테스트"""
        # MIDI 파일 벡터화
        vectorstore = self.vectorizer.vectorize_midi(self.test_midi_paths)
        
        # 벡터 저장소가 생성되었는지 확인
        self.assertIsNotNone(vectorstore)
        
        # 모든 MIDI 파일이 벡터화되었는지 확인
        self.assertEqual(len(vectorstore.docstore._dict), len(self.test_midi_paths))
    
    def test_save_and_load_vectorstore(self):
        """벡터 저장소 저장 및 로드 테스트"""
        # MIDI 파일 벡터화
        vectorstore = self.vectorizer.vectorize_midi(self.test_midi_paths)
        
        # 벡터 저장소 저장
        save_result = self.vectorizer.save_vectorstore(vectorstore, self.test_vectorstore_path)
        self.assertTrue(save_result)
        
        # 벡터 저장소 로드
        loaded_vectorstore = self.vectorizer.load_vectorstore(self.test_vectorstore_path)
        self.assertIsNotNone(loaded_vectorstore)
        
        # 원본과 로드된 벡터 저장소의 문서 수가 동일한지 확인
        self.assertEqual(len(vectorstore.docstore._dict), len(loaded_vectorstore.docstore._dict))
    
    def test_create_document(self):
        """Document 객체 생성 테스트"""
        # 테스트용 특징 데이터
        features = {
            'tempo': {'main_tempo': 120, 'tempo_changes': 1},
            'key_signatures': ['C major'],
            'time_signatures': ['4/4'],
            'instruments': ['Piano'],
            'pitch_range': [60, 72],
            'avg_note_duration': 0.5,
            'avg_velocity': 64
        }
        
        # Document 객체 생성 (비공개 메소드 직접 호출)
        doc = self.vectorizer._create_document(features, "test.mid")
        
        # Document 객체가 올바르게 생성되었는지 확인
        self.assertIsNotNone(doc)
        self.assertEqual(doc.metadata['filename'], "test.mid")
        self.assertIn("features", doc.metadata)
        self.assertIn("파일명: test.mid", doc.page_content)

if __name__ == '__main__':
    unittest.main()
