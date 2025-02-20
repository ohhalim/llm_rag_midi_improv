# llm_api.py
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate

# llm_api.py
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate

class LLMAPI:
    def __init__(self):
        self.llm = OllamaLLM(model="llama3.2")
        self.prompt = PromptTemplate.from_template("""
        다음 MIDI 파일의 특징을 반영한 새로운 MIDI를 생성하세요:

        입력 MIDI 특징:
        {input_features}
        
        유사한 MIDI 특징:
        {similar_features}
        
        위 특징들을 기반으로 다음 조건을 만족하는 JSON을 생성하세요:
        1. 최소 100개 이상의 음표
        2. 음높이는 33에서 85 사이
        3. 평균 음표 길이는 1.2초와 비슷하게
        4. 시작 시간은 이전 음표의 (시작 시간 + 지속 시간)으로 계산
        5. velocity는 32에서 127 사이의 다양한 값
        
        다음 JSON 형식으로만 응답하세요:
        {{
            "tracks": [
                {{
                    "instrument": 0,
                    "notes": [
                        {{"pitch": <33-85 사이>, "time": <누적 시간>, "duration": <평균 1.2초>, "velocity": <32-127>}},
                        ... (최소 100개)
                    ]
                }}
            ],
            "time_signatures": ["4/4"],
            "key_signatures": []
        }}
        
        반드시 위 JSON 형식으로만 응답하고, 다른 설명이나 코드는 포함하지 마세요.
        """)
    
    def generate_response(self, input_features, similar_features):
        """LLM을 사용하여 응답 생성"""
        prompt_value = self.prompt.format(
            input_features=input_features,
            similar_features=similar_features
        )
        
        response = self.llm.invoke(prompt_value)
        return response