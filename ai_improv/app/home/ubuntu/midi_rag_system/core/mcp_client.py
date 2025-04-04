"""
MCP 클라이언트 모듈
AI 모델이 MCP 서버와 통신하기 위한 클라이언트를 구현합니다.
"""
import json
from typing import Dict, List, Any, Optional
from mcp_python import MCPClient
import config

class MIDIMCPClient:
    def __init__(self, host=None, port=None):
        """
        MCP 클라이언트 초기화
        
        Args:
            host: 서버 호스트 (기본값: config.py에서 설정)
            port: 서버 포트 (기본값: config.py에서 설정)
        """
        self.host = host or config.MCP_SERVER_HOST
        self.port = port or config.MCP_SERVER_PORT
        
        # MCP 클라이언트 초기화
        self.client = MCPClient(host=self.host, port=self.port)
    
    async def search_midi(self, query: str, k: int = 3) -> Dict[str, Any]:
        """
        MIDI 검색 요청
        
        Args:
            query: 검색 쿼리
            k: 반환할 결과 수
            
        Returns:
            Dict: 검색 결과
        """
        try:
            response = await self.client.request(
                "/midi/search",
                json={"query": query, "k": k}
            )
            return response
        except Exception as e:
            print(f"MIDI 검색 중 오류 발생: {str(e)}")
            return {"error": str(e)}
    
    async def list_midi(self) -> Dict[str, Any]:
        """
        MIDI 파일 목록 요청
        
        Returns:
            Dict: MIDI 파일 목록
        """
        try:
            response = await self.client.request("/midi/list")
            return response
        except Exception as e:
            print(f"MIDI 파일 목록 조회 중 오류 발생: {str(e)}")
            return {"error": str(e)}
    
    async def get_midi_features(self, midi_file: str) -> Dict[str, Any]:
        """
        MIDI 특징 조회 요청
        
        Args:
            midi_file: MIDI 파일 경로
            
        Returns:
            Dict: MIDI 특징 정보
        """
        try:
            response = await self.client.request(
                "/midi/features",
                json={"midi_file": midi_file}
            )
            return response
        except Exception as e:
            print(f"MIDI 특징 조회 중 오류 발생: {str(e)}")
            return {"error": str(e)}
    
    async def close(self):
        """MCP 클라이언트 연결 종료"""
        await self.client.close()
