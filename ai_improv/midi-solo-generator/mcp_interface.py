# mcp_interface.py
import socket
import struct
import time
from typing import Optional, Dict, Any

class MCPInterface:
    """FL Studio와 통신하기 위한 MIDI Control Protocol 인터페이스"""
    
    def __init__(self, host: str = "127.0.0.1", port: int = 8080):
        self.host = host
        self.port = port
        self.socket = None
        self.connected = False
        
    def connect(self) -> bool:
        """FL Studio MCP 서버에 연결"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((self.host, self.port))
            self.connected = True
            return True
        except Exception as e:
            print(f"FL Studio 연결 실패: {str(e)}")
            self.connected = False
            return False
    
    def disconnect(self):
        """FL Studio 연결 해제"""
        if self.socket:
            self.socket.close()
        self.connected = False
    
    def send_midi(self, midi_data: bytes) -> bool:
        """
        MIDI 데이터를 FL Studio로 전송
        
        Args:
            midi_data: MIDI 파일 데이터
            
        Returns:
            성공 여부
        """
        if not self.connected:
            if not self.connect():
                return False
        
        try:
            # MCP 프로토콜 포맷 (헤더 + 데이터)
            # 헤더 형식: 'MCPF' + 길이(4바이트) + 명령어(1바이트)
            command_type = 1  # 1 = MIDI 데이터 전송
            header = b'MCPF' + struct.pack("<II", len(midi_data), command_type)
            
            # 헤더와 MIDI 데이터 전송
            self.socket.sendall(header + midi_data)
            
            # 응답 받기
            response = self.socket.recv(8)
            if len(response) >= 8 and response.startswith(b'MCPR'):
                status_code = struct.unpack("<I", response[4:8])[0]
                return status_code == 0  # 0 = 성공
            
            return False
            
        except Exception as e:
            print(f"MIDI 전송 실패: {str(e)}")
            self.connected = False
            return False
    
    def request_playback(self, start: bool = True) -> bool:
        """
        FL Studio 재생 시작/중지 요청
        
        Args:
            start: True = 재생 시작, False = 재생 중지
            
        Returns:
            성공 여부
        """
        if not self.connected:
            if not self.connect():
                return False
        
        try:
            # 재생 제어 명령
            command_type = 2  # 2 = 트랜스포트 제어
            playback_command = 1 if start else 0  # 1 = 재생, 0 = 중지
            data = struct.pack("<I", playback_command)
            
            header = b'MCPF' + struct.pack("<II", len(data), command_type)
            
            # 명령 전송
            self.socket.sendall(header + data)
            
            # 응답 받기
            response = self.socket.recv(8)
            if len(response) >= 8 and response.startswith(b'MCPR'):
                status_code = struct.unpack("<I", response[4:8])[0]
                return status_code == 0  # 0 = 성공
            
            return False
            
        except Exception as e:
            print(f"재생 제어 실패: {str(e)}")
            self.connected = False
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """
        FL Studio 상태 정보 가져오기
        
        Returns:
            상태 정보 딕셔너리
        """
        status = {
            "connected": self.connected,
            "host": self.host,
            "port": self.port
        }
        
        if not self.connected:
            return status
        
        try:
            # 상태 요청 명령
            command_type = 3  # 3 = 상태 요청
            header = b'MCPF' + struct.pack("<II", 0, command_type)
            
            # 명령 전송
            self.socket.sendall(header)
            
            # 응답 헤더 받기
            response_header = self.socket.recv(12)
            if len(response_header) >= 12 and response_header.startswith(b'MCPR'):
                data_length = struct.unpack("<I", response_header[4:8])[0]
                status_code = struct.unpack("<I", response_header[8:12])[0]
                
                if status_code == 0 and data_length > 0:
                    # 상태 데이터 받기
                    status_data = self.socket.recv(data_length)
                    
                    # 간단한 파싱 (실제로는 더 복잡할 수 있음)
                    playback_status = status_data[0] if len(status_data) > 0 else 0
                    tempo = struct.unpack("<f", status_data[1:5])[0] if len(status_data) > 4 else 120.0
                    
                    status.update({
                        "playback_status": "playing" if playback_status == 1 else "stopped",
                        "tempo": tempo
                    })
            
        except Exception as e:
            print(f"상태 요청 실패: {str(e)}")
            self.connected = False
        
        return status