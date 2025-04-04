"""
YuE 음악 생성 모듈 테스트
YuEMusicGenerator 클래스의 기능을 테스트합니다.
"""
import unittest
import os
import sys
import io
from pathlib import Path
from unittest.mock import MagicMock, patch

# 프로젝트 루트 디렉토리를 Python 경로에 추가
sys.path.append(str(Path(__file__).resolve().parent.parent))

from midi_rag_system.models.yue_generator import YuEMusicGenerator

class TestYuEMusicGenerator(unittest.TestCase):
    """YuEMusicGenerator 테스트 클래스"""
    
    def setUp(self):
        """테스트 설정"""
        # 모의 모델을 사용하는 YuEMusicGenerator 인스턴스 생성
        self.generator = self._create_test_generator()
        
        # 테스트용 출력 파일 경로
        self.test_output_path = os.path.join(os.path.dirname(__file__), "test_output.mid")
    
    def tearDown(self):
        """테스트 정리"""
        # 테스트용 출력 파일 삭제
        if os.path.exists(self.test_output_path):
            os.remove(self.test_output_path)
    
    def _create_test_generator(self):
        """테스트용 YuE 생성기 생성 (모의 모델 사용)"""
        # 실제 YuEMusicGenerator 인스턴스 생성
        generator = YuEMusicGenerator()
        
        # 모델과 토크나이저를 모의 객체로 대체
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {"input_ids": MagicMock(), "attention_mask": MagicMock()}
        mock_tokenizer.decode.return_value = "테스트 출력 텍스트"
        
        mock_model = MagicMock()
        mock_model.generate.return_value = [MagicMock()]
        mock_model.device = "cpu"
        
        generator.stage1_tokenizer = mock_tokenizer
        generator.stage1_model = mock_model
        generator.stage2_tokenizer = mock_tokenizer
        generator.stage2_model = mock_model
        
        return generator
    
    def test_generate_music(self):
        """음악 생성 테스트"""
        # _load_models 메소드를 모의 함수로 대체
        with patch.object(self.generator, '_load_models', return_value=None):
            # 음악 생성
            midi_data = self.generator.generate_music(
                genre_text="장르: 팝, 스타일: 현대적",
                lyrics_text="테스트 가사",
                run_n_segments=1,
                stage2_batch_size=1
            )
            
            # MIDI 데이터가 생성되었는지 확인
            self.assertIsNotNone(midi_data)
            self.assertIsInstance(midi_data, bytes)
    
    def test_generate_sample_midi_json(self):
        """샘플 MIDI JSON 생성 테스트"""
        # 샘플 MIDI JSON 생성
        midi_data = self.generator._generate_sample_midi_json()
        
        # JSON 데이터가 올바른 형식인지 확인
        self.assertIn("tracks", midi_data)
        self.assertIn("time_signatures", midi_data)
        self.assertIn("key_signatures", midi_data)
        
        # 트랙에 노트가 포함되어 있는지 확인
        self.assertGreater(len(midi_data["tracks"]), 0)
        self.assertGreater(len(midi_data["tracks"][0]["notes"]), 0)
        
        # 노트가 올바른 형식인지 확인
        note = midi_data["tracks"][0]["notes"][0]
        self.assertIn("pitch", note)
        self.assertIn("time", note)
        self.assertIn("duration", note)
        self.assertIn("velocity", note)
    
    def test_convert_json_to_midi(self):
        """JSON을 MIDI로 변환 테스트"""
        # 테스트용 JSON 데이터
        json_data = {
            "tracks": [
                {
                    "instrument": 0,
                    "notes": [
                        {"pitch": 60, "time": 0.0, "duration": 0.5, "velocity": 64},
                        {"pitch": 62, "time": 0.5, "duration": 0.5, "velocity": 64},
                        {"pitch": 64, "time": 1.0, "duration": 0.5, "velocity": 64}
                    ]
                }
            ],
            "time_signatures": ["4/4"],
            "key_signatures": []
        }
        
        # JSON을 MIDI로 변환
        midi_data = self.generator._convert_json_to_midi(json_data)
        
        # MIDI 데이터가 생성되었는지 확인
        self.assertIsNotNone(midi_data)
        self.assertIsInstance(midi_data, bytes)
        
        # MIDI 파일 형식인지 확인 (MThd로 시작하는지)
        self.assertTrue(midi_data.startswith(b'MThd'))
    
    def test_save_midi(self):
        """MIDI 저장 테스트"""
        # 테스트용 MIDI 데이터
        midi_data = b'MThd' + b'\x00\x00\x00\x06\x00\x01\x00\x02\x01\xe0'  # 최소한의 MIDI 헤더
        
        # MIDI 저장
        result = self.generator.save_midi(midi_data, self.test_output_path)
        
        # 저장 성공 여부 확인
        self.assertTrue(result)
        
        # 파일이 생성되었는지 확인
        self.assertTrue(os.path.exists(self.test_output_path))
        
        # 파일 내용이 올바른지 확인
        with open(self.test_output_path, 'rb') as f:
            saved_data = f.read()
        self.assertEqual(saved_data, midi_data)

if __name__ == '__main__':
    unittest.main()
