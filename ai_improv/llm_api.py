import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

class LLMAPI:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4")
        self.prompt = PromptTemplate.from_template("""
        입력된 MIDI 파일의 특징:
        {input_features}
        
        유사한 MIDI 파일들의 특징:
        {similar_features}
        
        위 특징들을 기반으로 새로운 MIDI 파일을 생성하세요.
        """)
    
    def generate_response(self, input_features, similar_features):
        # LLM 응답 생성
        pass