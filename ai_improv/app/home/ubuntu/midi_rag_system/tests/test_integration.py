"""
통합 테스트 모듈
MIDI RAG 시스템의 전체 기능을 테스트합니다.
"""
import unittest
import os
import sys
import asyncio
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

# 프로젝트 루트 디렉토리를 Python 경로에 추가
sys.path.append(str(Path(__file__).resolve().parent.parent))

from midi_rag_system.core.midi_rag_system import MIDIRAGSystem
import config

class TestMIDIRAGSystem(unittest.TestCase):
    """MIDIRAGSystem 통합 테스트 클래스"""
    
    def setUp(self):
        """테스트 설정"""
        # 테스트용 MIDI RAG 시스템 생성
        self.rag_system = self._create_test_rag_system()
        
        # 테스트용 MIDI 파일 생성
        self.test_midi_paths = self._create_test_midi_files(3)
        self.test_input_midi = self.test_midi_paths[0]
        
        # 테스트용 벡터 저장소 경로
        self.test_vectorstore_path = tempfile.mkdtemp()
        
        # 테스트용 출력 파일 경로
        self.test_output_path = os.path.join(os.path.dirname(__file__), "test_output.mid")
    
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
        
        # 테스트용 출력 파일 삭제
        if os.path.exists(self.test_output_path):
            os.remove(self.test_output_path)
    
    def _create_test_rag_system(self):
        """테스트용 MIDI RAG 시스템 생성 (모의 객체 사용)"""
        # 실제 MIDIRAGSystem 인스턴스 생성
        rag_system = MIDIRAGSystem()
        
        # 각 구성 요소를 모의 객체로 대체
        mock_vectorizer = MagicMock()
        mock_vectorizer.vectorize_midi.return_value = MagicMock()
        mock_vectorizer.save_vectorstore.return_value = True
        mock_vectorizer.load_vectorstore.return_value = MagicMock()
        
        mock_feature_extractor = MagicMock()
        mock_feature_extractor.extract_features.return_value = {
            "tempo": {"main_tempo": 120, "tempo_changes": 1},
            "key_signatures": ["C major"],
            "time_signatures": ["4/4"],
            "instruments": ["Piano"]
        }
        
        mock_llm_api = MagicMock()
        mock_llm_api.generate_response.return_value = '{"test": "data"}'
        
        mock_yue_generator = MagicMock()
        mock_yue_generator.generate_music.return_value = b'MThd' + b'\x00\x00\x00\x06\x00\x01\x00\x02\x01\xe0'
        
        rag_system.vectorizer = mock_vectorizer
        rag_system.feature_extractor = mock_feature_extractor
        rag_system.llm_api = mock_llm_api
        rag_system.yue_generator = mock_yue_generator
        rag_system.vectorstore = MagicMock()
        
        return rag_system
    
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
    
    def test_train(self):
        """학습 기능 테스트"""
        # 학습 실행
        self.rag_system.train(self.test_midi_paths, self.test_vectorstore_path)
        
        # vectorize_midi 메소드가 호출되었는지 확인
        self.rag_system.vectorizer.vectorize_midi.assert_called_once_with(self.test_midi_paths)
        
        # save_vectorstore 메소드가 호출되었는지 확인
        self.rag_system.vectorizer.save_vectorstore.assert_called_once()
    
    def test_load_vectorstore(self):
        """벡터 저장소 로드 테스트"""
        # 벡터 저장소 로드
        result = self.rag_system.load_vectorstore(self.test_vectorstore_path)
        
        # 로드 성공 여부 확인
        self.assertTrue(result)
        
        # load_vectorstore 메소드가 호출되었는지 확인
        self.rag_system.vectorizer.load_vectorstore.assert_called_once_with(self.test_vectorstore_path)
    
    def test_generate(self):
        """MIDI 생성 테스트"""
        # 비동기 테스트를 위한 래퍼 함수
        async def async_test():
            # MIDI 생성
            midi_data = await self.rag_system.generate(
                self.test_input_midi,
                output_format='midi',
                genre_text="장르: 팝, 스타일: 현대적",
                lyrics_text="테스트 가사"
            )
            
            # MIDI 데이터가 생성되었는지 확인
            self.assertIsNotNone(midi_data)
            self.assertIsInstance(midi_data, bytes)
            
            # extract_features 메소드가 호출되었는지 확인
            self.rag_system.feature_extractor.extract_features.assert_called_once_with(self.test_input_midi)
            
            # similarity_search 메소드가 호출되었는지 확인
            self.rag_system.vectorstore.similarity_search.assert_called_once()
            
            # generate_music 메소드가 호출되었는지 확인
            self.rag_system.yue_generator.generate_music.assert_called_once()
        
        # 비동기 테스트 실행
        asyncio.run(async_test())
    
    def test_save_midi(self):
        """MIDI 저장 테스트"""
        # 테스트용 MIDI 데이터
        midi_data = b'MThd' + b'\x00\x00\x00\x06\x00\x01\x00\x02\x01\xe0'  # 최소한의 MIDI 헤더
        
        # MIDI 저장
        self.rag_system.save_midi(midi_data, self.test_output_path)
        
        # 파일이 생성되었는지 확인
        self.assertTrue(os.path.exists(self.test_output_path))
        
        # 파일 내용이 올바른지 확인
        with open(self.test_output_path, 'rb') as f:
            saved_data = f.read()
        self.assertEqual(saved_data, midi_data)
    
    def test_end_to_end_workflow(self):
        """전체 워크플로우 테스트"""
        # 비동기 테스트를 위한 래퍼 함수
        async def async_test():
            # 1. 학습
            self.rag_system.train(self.test_midi_paths, self.test_vectorstore_path)
            
            # 2. MIDI 생성
            midi_data = await self.rag_system.generate(
                self.test_input_midi,
                output_format='midi'
            )
            
            # 3. MIDI 저장
            self.rag_system.save_midi(midi_data, self.test_output_path)
            
            # 각 단계가 성공적으로 완료되었는지 확인
            self.rag_system.vectorizer.vectorize_midi.assert_called_once()
            self.rag_system.feature_extractor.extract_features.assert_called_once()
            self.rag_system.vectorstore.similarity_search.assert_called_once()
            self.rag_system.yue_generator.generate_music.assert_called_once()
            self.assertTrue(os.path.exists(self.test_output_path))
        
        # 비동기 테스트 실행
        asyncio.run(async_test())

if __name__ == '__main__':
    unittest.main()
