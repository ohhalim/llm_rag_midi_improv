"""
LLM API 모듈 테스트
LLMAPI 클래스의 기능을 테스트합니다.
"""
import unittest
import os
import sys
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

# 프로젝트 루트 디렉토리를 Python 경로에 추가
sys.path.append(str(Path(__file__).resolve().parent.parent))

from midi_rag_system.core.llm_api import LLMAPI

class TestLLMAPI(unittest.TestCase):
    """LLMAPI 테스트 클래스"""
    
    def setUp(self):
        """테스트 설정"""
        # 모의 LLM 모델을 사용하는 LLMAPI 인스턴스 생성
        self.llm_api = self._create_test_llm_api()
    
    def _create_test_llm_api(self):
        """테스트용 LLM API 모듈 생성 (모의 LLM 사용)"""
        # 실제 LLMAPI 인스턴스 생성
        llm_api = LLMAPI()
        
        # LLM 모델을 모의 객체로 대체
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = """
        다음은 요청하신 MIDI 데이터입니다:
        
        ```json
        {
            "tracks": [
                {
                    "instrument": 0,
                    "notes": [
                        {"pitch": 60, "time": 0.0, "duration": 1.2, "velocity": 64},
                        {"pitch": 62, "time": 1.2, "duration": 1.2, "velocity": 64},
                        {"pitch": 64, "time": 2.4, "duration": 1.2, "velocity": 64}
                    ]
                }
            ],
            "time_signatures": ["4/4"],
            "key_signatures": []
        }
        ```
        """
        
        llm_api.llm = mock_llm
        
        return llm_api
    
    def test_clean_llm_response(self):
        """LLM 응답 정제 테스트"""
        # 테스트용 LLM 응답
        test_response = """
        다음은 요청하신 MIDI 데이터입니다:
        
        ```json
        {
            "tracks": [
                {
                    "instrument": 0,
                    "notes": [
                        {"pitch": 60, "time": 0.0, "duration": 1.2, "velocity": 64},
                        {"pitch": 62, "time": 1.2, "duration": 1.2, "velocity": 64},
                        {"pitch": 64, "time": 2.4, "duration": 1.2, "velocity": 64}
                    ]
                }
            ],
            "time_signatures": ["4/4"],
            "key_signatures": []
        }
        ```
        """
        
        # 응답 정제
        cleaned_json = self.llm_api.clean_llm_response(test_response)
        
        # 정제된 JSON이 유효한지 확인
        json_data = json.loads(cleaned_json)
        
        # JSON 구조 확인
        self.assertIn("tracks", json_data)
        self.assertEqual(len(json_data["tracks"]), 1)
        self.assertEqual(len(json_data["tracks"][0]["notes"]), 3)
        self.assertEqual(json_data["tracks"][0]["notes"][0]["pitch"], 60)
    
    def test_clean_llm_response_with_expressions(self):
        """수식이 포함된 LLM 응답 정제 테스트"""
        # 수식이 포함된 테스트용 LLM 응답
        test_response = """
        ```json
        {
            "tracks": [
                {
                    "instrument": 0,
                    "notes": [
                        {"pitch": 60, "time": 0.0, "duration": 1.2, "velocity": 64},
                        {"pitch": 62, "time": 0.0 + 1.2, "duration": 1.2, "velocity": 64},
                        {"pitch": 64, "time": 1.2 + 1.2, "duration": 1.2, "velocity": 64}
                    ]
                }
            ],
            "time_signatures": ["4/4"],
            "key_signatures": []
        }
        ```
        """
        
        # 응답 정제
        cleaned_json = self.llm_api.clean_llm_response(test_response)
        
        # 정제된 JSON이 유효한지 확인
        json_data = json.loads(cleaned_json)
        
        # 수식이 계산되었는지 확인
        self.assertEqual(json_data["tracks"][0]["notes"][1]["time"], 1.2)
        self.assertEqual(json_data["tracks"][0]["notes"][2]["time"], 2.4)
    
    def test_enhance_solo(self):
        """솔로 확장 테스트"""
        # 짧은 솔로 JSON
        short_solo = json.dumps({
            "tracks": [
                {
                    "instrument": 0,
                    "notes": [
                        {"pitch": 60, "time": 0.0, "duration": 1.2, "velocity": 64},
                        {"pitch": 62, "time": 1.2, "duration": 1.2, "velocity": 64},
                        {"pitch": 64, "time": 2.4, "duration": 1.2, "velocity": 64}
                    ]
                }
            ],
            "time_signatures": ["4/4"],
            "key_signatures": []
        })
        
        # 솔로 확장
        enhanced_json = self.llm_api.enhance_solo(short_solo, min_notes=10)
        
        # 확장된 JSON이 유효한지 확인
        json_data = json.loads(enhanced_json)
        
        # 노트 수가 증가했는지 확인
        self.assertGreaterEqual(len(json_data["tracks"][0]["notes"]), 10)
    
    def test_generate_response(self):
        """응답 생성 테스트"""
        # 테스트용 입력 및 유사 특징
        input_features = "템포: 120, 조표: C major, 박자: 4/4"
        similar_features = "템포: 125, 조표: C major, 박자: 4/4"
        
        # 응답 생성
        with patch.object(self.llm_api, 'clean_llm_response', return_value='{"test": "data"}'):
            with patch.object(self.llm_api, 'enhance_solo', return_value='{"test": "enhanced"}'):
                response = self.llm_api.generate_response(input_features, similar_features)
        
        # 응답이 생성되었는지 확인
        self.assertEqual(response, '{"test": "enhanced"}')

if __name__ == '__main__':
    unittest.main()
