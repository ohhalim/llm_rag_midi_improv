from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
import json
import re
import ast
import math

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
        4. 시작 시간은 숫자로 직접 계산된 값으로 작성 (수식이 아닌 계산된 숫자값으로)
        5. velocity는 32에서 127 사이의 다양한 값
        
        다음 JSON 형식으로만 응답하세요:
        ```json
        {{
            "tracks": [
                {{
                    "instrument": 0,
                    "notes": [
                        {{"pitch": <33-85 사이>, "time": <시간(숫자)>, "duration": <평균 1.2초>, "velocity": <32-127>}},
                        ... (최소 100개)
                    ]
                }}
            ],
            "time_signatures": ["4/4"],
            "key_signatures": []
        }}
        ```
        노트는 반드시 100개 이상이어야 합니다.
        반드시 위 JSON 형식으로만 응답하고, 다른 설명이나 코드는 포함하지 마세요.
        시간 값은 절대 수식으로 표현하지 말고 계산된 실제 숫자로만 작성하세요. (예: "time": 2.4 (O), "time": "0.6 + 1.8" (X))
        """)
    
    def clean_llm_response(self, response):
        """LLM 응답에서 JSON 부분만 추출하고 수식을 계산합니다."""
        
        # JSON 부분 추출 (```json과 ``` 사이의 내용)
        json_pattern = r"```json\s*([\s\S]*?)\s*```"
        json_match = re.search(json_pattern, response)
        
        if json_match:
            json_str = json_match.group(1)
        else:
            # JSON 코드 블록이 없으면 중괄호로 둘러싸인 부분 찾기
            json_pattern = r"\{\s*[\s\S]*?\s*\}"
            json_match = re.search(json_pattern, response)
            if json_match:
                json_str = json_match.group(0)
            else:
                raise ValueError("응답에서 JSON을 찾을 수 없습니다.")
        
        # 수식 계산 함수
        def evaluate_expressions(match):
            expr = match.group(1)
            try:
                # 안전하게 수식 계산 (기본 연산자만 허용)
                calculated = ast.literal_eval(expr.replace('^', '**'))
                return str(calculated)
            except:
                return match.group(0)
        
        # 수식 패턴: 숫자와 기본 연산자(+, -, *, /, ^)로 구성된 표현식
        expression_pattern = r"([\d.]+\s*[\+\-\*\/\^]\s*[\(\)\d\s\+\-\*\/\^\.]*)"
        json_str = re.sub(expression_pattern, evaluate_expressions, json_str)
        
        # 중첩된 표현식 처리 (예: "time": 0.6 + (1.8*2))
        def fix_expr_in_json(json_str):
            # 키-값 패턴에서 값이 수식인 경우 찾기
            pattern = r'("[\w]+")\s*:\s*([\d.]+\s*[\+\-\*\/\^]\s*[\(\)\d\s\+\-\*\/\^\.]*)(,|\s*\})'
            
            def eval_and_replace(match):
                key = match.group(1)
                expr = match.group(2)
                end = match.group(3)
                
                try:
                    # 제한된 환경에서 수식 계산
                    result = eval(expr, {"__builtins__": {}}, {"math": math})
                    return f"{key}: {result}{end}"
                except:
                    return match.group(0)
            
            result = re.sub(pattern, eval_and_replace, json_str)
            # 변경된 것이 있으면 재귀적으로 처리 (중첩된 표현식 처리)
            if result != json_str:
                return fix_expr_in_json(result)
            return result
        
        json_str = fix_expr_in_json(json_str)
        
        try:
            # JSON 유효성 검사 및 파싱
            data = json.loads(json_str)
            
            # tracks의 notes에서 수식 계산 (중첩된 객체 처리)
            if 'tracks' in data:
                for track in data['tracks']:
                    if 'notes' in track:
                        for note in track['notes']:
                            for key, value in note.items():
                                if isinstance(value, str):
                                    try:
                                        # 문자열이 수식인 경우 계산
                                        note[key] = eval(value, {"__builtins__": {}}, {"math": math})
                                    except:
                                        pass
            
            # 다시 JSON 문자열로 변환
            return json.dumps(data, indent=2, ensure_ascii=False)
        except json.JSONDecodeError as e:
            raise ValueError(f"JSON 파싱 오류: {str(e)}\n원본 JSON: {json_str}")
    
    def generate_response(self, input_features, similar_features):
        """LLM을 사용하여 응답 생성하고 정제된 JSON 반환"""
        prompt_value = self.prompt.format(
            input_features=input_features,
            similar_features=similar_features
        )
        
        response = self.llm.invoke(prompt_value)
        
        try:
            # 응답에서 JSON 추출 및 정제
            cleaned_json = self.clean_llm_response(response)
            return cleaned_json
        except Exception as e:
            print(f"응답 정제 중 오류 발생: {str(e)}")
            print(f"원본 응답: {response}")
            # 오류 발생 시 원본 응답 반환
            return response