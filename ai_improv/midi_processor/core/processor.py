"""
MIDI 프로세서 모듈
MIDI 파일 로드, 처리 및 저장 기능을 제공하는 주요 클래스
"""
import os
import logging
from ..utils.midi_utils import (
    load_midi_file, 
    save_midi_file, 
    midi_to_json, 
    json_to_midi
)

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MIDIProcessor:
    """
    MIDI 파일 처리 클래스
    MIDI 파일 로드, 변환, 저장 기능 제공
    """
    
    def __init__(self):
        """초기화"""
        self.midi_file = None
        self.json_data = None
    
    def load(self, file_path):
        """
        MIDI 파일 로드
        
        Args:
            file_path: MIDI 파일 경로
            
        Returns:
            bool: 성공 여부
        """
        try:
            self.midi_file = load_midi_file(file_path)
            if self.midi_file:
                logger.info(f"MIDI 파일을 성공적으로 로드했습니다: {file_path}")
                return True
            return False
        except Exception as e:
            logger.error(f"MIDI 파일 로드 중 오류: {str(e)}")
            return False
    
    def save(self, output_path):
        """
        MIDI 파일 저장
        
        Args:
            output_path: 저장할 파일 경로
            
        Returns:
            bool: 성공 여부
        """
        if not self.midi_file:
            logger.error("저장할 MIDI 파일이 없습니다. 먼저 파일을 로드하세요.")
            return False
        
        # 디렉토리 확인 및 생성
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        return save_midi_file(self.midi_file, output_path)
    
    def to_json(self):
        """
        MIDI 파일을 JSON 형식으로 변환
        
        Returns:
            str: JSON 문자열
        """
        if not self.midi_file:
            logger.error("변환할 MIDI 파일이 없습니다. 먼저 파일을 로드하세요.")
            return "{}"
        
        self.json_data = midi_to_json(self.midi_file)
        return self.json_data
    
    def from_json(self, json_string):
        """
        JSON 문자열을 MIDI 파일로 변환
        
        Args:
            json_string: JSON 문자열
            
        Returns:
            bool: 성공 여부
        """
        try:
            self.midi_file = json_to_midi(json_string)
            if self.midi_file:
                logger.info("JSON을 MIDI로 성공적으로 변환했습니다.")
                return True
            logger.error("JSON을 MIDI로 변환하지 못했습니다.")
            return False
        except Exception as e:
            logger.error(f"JSON을 MIDI로 변환 중 오류: {str(e)}")
            return False
    
    def get_info(self):
        """
        MIDI 파일 정보 반환
        
        Returns:
            dict: MIDI 파일 정보
        """
        if not self.midi_file:
            logger.error("MIDI 파일이 로드되지 않았습니다.")
            return {}
        
        info = {
            "ticks_per_beat": self.midi_file.ticks_per_beat,
            "track_count": len(self.midi_file.tracks),
            "tracks": []
        }
        
        for i, track in enumerate(self.midi_file.tracks):
            track_info = {
                "index": i,
                "name": track.name if hasattr(track, 'name') else f"Track {i}",
                "event_count": len(track),
                "note_count": sum(1 for msg in track if msg.type in ['note_on', 'note_off'])
            }
            info["tracks"].append(track_info)
        
        return info 