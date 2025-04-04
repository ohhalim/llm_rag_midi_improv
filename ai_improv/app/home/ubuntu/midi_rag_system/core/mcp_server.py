"""
MCP(Model Context Protocol) 서버 모듈
MIDI 데이터 소스와 AI 모델 간의 연결을 위한 MCP 서버를 구현합니다.
"""
import os
import json
from typing import Dict, List, Any, Optional
from mcp_python import MCPServer, MCPContext
import config
from midi_rag_system.core.midi_vectorizer import MIDIVectorizer

class MIDIMCPServer:
    def __init__(self, host=None, port=None, vectorstore_path=None):
        """
        MCP 서버 초기화
        
        Args:
            host: 서버 호스트 (기본값: config.py에서 설정)
            port: 서버 포트 (기본값: config.py에서 설정)
            vectorstore_path: 벡터 저장소 경로 (기본값: config.py에서 설정)
        """
        self.host = host or config.MCP_SERVER_HOST
        self.port = port or config.MCP_SERVER_PORT
        self.vectorstore_path = vectorstore_path or config.VECTORSTORE_PATH
        
        # 벡터 저장소 초기화
        self.vectorizer = MIDIVectorizer()
        self.vectorstore = None
        
        # MCP 서버 초기화
        self.server = MCPServer(host=self.host, port=self.port)
        
        # 서버 라우트 등록
        self._register_routes()
    
    def _register_routes(self):
        """MCP 서버 라우트 등록"""
        self.server.register_route("/midi/search", self.search_midi)
        self.server.register_route("/midi/list", self.list_midi)
        self.server.register_route("/midi/features", self.get_midi_features)
    
    def start(self):
        """MCP 서버 시작"""
        # 벡터 저장소 로드
        self._load_vectorstore()
        
        # 서버 시작
        print(f"MCP 서버 시작: {self.host}:{self.port}")
        self.server.start()
    
    def stop(self):
        """MCP 서버 중지"""
        self.server.stop()
        print("MCP 서버 중지됨")
    
    def _load_vectorstore(self):
        """벡터 저장소 로드"""
        if os.path.exists(self.vectorstore_path):
            self.vectorstore = self.vectorizer.load_vectorstore(self.vectorstore_path)
            if self.vectorstore:
                print(f"벡터 저장소 로드 완료: {self.vectorstore_path}")
            else:
                print(f"벡터 저장소 로드 실패: {self.vectorstore_path}")
        else:
            print(f"벡터 저장소가 존재하지 않습니다: {self.vectorstore_path}")
    
    async def search_midi(self, context: MCPContext) -> Dict[str, Any]:
        """
        MIDI 검색 엔드포인트
        
        Args:
            context: MCP 컨텍스트
            
        Returns:
            Dict: 검색 결과
        """
        if not self.vectorstore:
            return {"error": "벡터 저장소가 로드되지 않았습니다."}
        
        # 요청 파라미터 파싱
        params = await context.json()
        query = params.get("query", "")
        k = params.get("k", 3)
        
        # 유사도 검색 수행
        try:
            similar_docs = self.vectorstore.similarity_search(query, k=k)
            results = []
            
            for i, doc in enumerate(similar_docs):
                results.append({
                    "index": i,
                    "filename": doc.metadata.get("filename", "Unknown"),
                    "content": doc.page_content,
                    "metadata": doc.metadata
                })
            
            return {
                "query": query,
                "results": results,
                "count": len(results)
            }
        except Exception as e:
            return {"error": f"검색 중 오류 발생: {str(e)}"}
    
    async def list_midi(self, context: MCPContext) -> Dict[str, Any]:
        """
        MIDI 파일 목록 엔드포인트
        
        Args:
            context: MCP 컨텍스트
            
        Returns:
            Dict: MIDI 파일 목록
        """
        try:
            # 학습 디렉토리에서 MIDI 파일 목록 가져오기
            midi_files = []
            training_dir = config.TRAINING_DIR
            
            if os.path.exists(training_dir):
                for file in os.listdir(training_dir):
                    if file.lower().endswith(('.mid', '.midi')):
                        midi_files.append(os.path.join(training_dir, file))
            
            return {
                "midi_files": midi_files,
                "count": len(midi_files)
            }
        except Exception as e:
            return {"error": f"MIDI 파일 목록 조회 중 오류 발생: {str(e)}"}
    
    async def get_midi_features(self, context: MCPContext) -> Dict[str, Any]:
        """
        MIDI 특징 조회 엔드포인트
        
        Args:
            context: MCP 컨텍스트
            
        Returns:
            Dict: MIDI 특징 정보
        """
        # 요청 파라미터 파싱
        params = await context.json()
        midi_file = params.get("midi_file", "")
        
        if not midi_file or not os.path.exists(midi_file):
            return {"error": f"MIDI 파일을 찾을 수 없습니다: {midi_file}"}
        
        try:
            # MIDI 특징 추출
            features = self.vectorizer.feature_extractor.extract_features(midi_file)
            
            return {
                "midi_file": midi_file,
                "features": features
            }
        except Exception as e:
            return {"error": f"MIDI 특징 추출 중 오류 발생: {str(e)}"}
