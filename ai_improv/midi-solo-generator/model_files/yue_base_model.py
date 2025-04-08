import torch
import numpy as np
import json
from typing import Dict, List, Any
import os

class YueBaseModel:
    """
    경량화된 YuE 모델 - MIDI 솔로 즉흥연주 생성용 부분 모델
    """
    def __init__(self):
        # 모델 메타데이터
        self.metadata = {
            "name": "YuE-Solo-Generator-Light",
            "version": "0.1.0",
            "description": "MIDI 솔로 라인 생성을 위해 경량화된 YuE 모델",
            "architecture": "부분 모델 (코드 분석 및 스타일 변환 컴포넌트)",
            "original_model": "YuE-7B",
            "compression": "부분 모델 추출",
            "input_type": "MIDI",
            "output_type": "MIDI 컨텍스트",
            "supported_styles": ["jazz_funk", "bebop", "blues", "fusion"]
        }
        
        # 모델 컴포넌트 로드
        self._load_components()
    
    def _load_components(self):
        """모델 컴포넌트 로드"""
        # 코드 분석 컴포넌트
        self.chord_analyzer = self._load_chord_analyzer()
        
        # 멜로디 스타일 특성
        self.style_characteristics = self._load_style_characteristics()
        
        # 스케일-코드 매핑
        self.scale_chord_mapping = self._load_scale_chord_mapping()
    
    def _load_chord_analyzer(self) -> Dict:
        """코드 분석 컴포넌트 로드"""
        # 간소화된 화성 분석 모듈
        chord_analyzer = {
            # 코드 루트 검출을 위한 가중치
            "root_detection_weights": [8.0, 0.5, 1.0, 0.5, 3.0, 2.0, 0.5, 5.0, 0.5, 1.0, 0.5, 2.0],
            
            # 코드 유형 분류를 위한 템플릿
            "chord_templates": {
                "maj": [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
                "min": [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
                "7": [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
                "maj7": [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1],
                "m7": [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0],
                "dim": [1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0],
                "aug": [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
                "sus4": [1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0]
            },
            
            # 키 감지를 위한 가중치
            "key_detection_weights": [
                [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88],  # Major
                [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]   # Minor
            ]
        }
        
        return chord_analyzer
    
    def _load_style_characteristics(self) -> Dict:
        """스타일별 음악적 특성 로드"""
        # 각 스타일의 음악적 특성
        style_characteristics = {
            "jazz_funk": {
                "rhythm_patterns": {
                    "0.25": 0.6,  # 16분음표
                    "0.5": 0.35,  # 8분음표
                    "1.0": 0.05   # 4분음표
                },
                "velocity_range": [70, 100],
                "articulation": 0.9,  # 스타카토 정도
                "scale_preference": "lydian_dominant",
                "chord_extensions": ["9", "13"],
                "syncopation": 0.7
            },
            "bebop": {
                "rhythm_patterns": {
                    "0.125": 0.3,  # 32분음표
                    "0.25": 0.5,   # 16분음표
                    "0.5": 0.2     # 8분음표
                },
                "velocity_range": [70, 110],
                "articulation": 0.8,  # 스타카토 정도
                "scale_preference": "bebop_dominant",
                "chord_extensions": ["7", "9", "b9"],
                "syncopation": 0.5
            },
            "blues": {
                "rhythm_patterns": {
                    "0.33": 0.3,  # 셋잇단음표
                    "0.5": 0.5,   # 8분음표
                    "1.0": 0.2    # 4분음표
                },
                "velocity_range": [75, 115],
                "articulation": 0.85,  # 스타카토 정도
                "scale_preference": "blues_scale",
                "chord_extensions": ["7", "9"],
                "syncopation": 0.6,
                "bend_probability": 0.4
            },
            "fusion": {
                "rhythm_patterns": {
                    "0.25": 0.4,   # 16분음표
                    "0.33": 0.3,   # 셋잇단음표
                    "0.5": 0.25,   # 8분음표
                    "1.0": 0.05    # 4분음표
                },
                "velocity_range": [65, 110],
                "articulation": 0.88,  # 스타카토 정도
                "scale_preference": "altered",
                "chord_extensions": ["7", "9", "11", "13"],
                "syncopation": 0.8
            }
        }
        
        return style_characteristics
    
    def _load_scale_chord_mapping(self) -> Dict:
        """스케일과 코드 간의 매핑 로드"""
        # 코드별 적합한 스케일 매핑
        scale_chord_mapping = {
            "maj": {
                "primary": [0, 2, 4, 5, 7, 9, 11],  # 메이저 스케일
                "secondary": [0, 2, 4, 6, 7, 9, 11]  # 리디안
            },
            "maj7": {
                "primary": [0, 2, 4, 5, 7, 9, 11],  # 메이저 스케일
                "secondary": [0, 2, 4, 6, 7, 9, 11]  # 리디안
            },
            "7": {
                "primary": [0, 2, 4, 5, 7, 9, 10],  # 믹솔리디안
                "secondary": [0, 2, 4, 5, 7, 9, 10, 11]  # 도미넌트 비밥
            },
            "min": {
                "primary": [0, 2, 3, 5, 7, 8, 10],  # 내츄럴 마이너
                "secondary": [0, 2, 3, 5, 7, 9, 10]  # 도리안
            },
            "m7": {
                "primary": [0, 2, 3, 5, 7, 9, 10],  # 도리안
                "secondary": [0, 2, 3, 5, 7, 8, 10]  # 내츄럴 마이너
            },
            "dim": {
                "primary": [0, 2, 3, 5, 6, 8, 9, 11],  # 디미니쉬드
                "secondary": [0, 1, 3, 4, 6, 7, 9, 10]  # 디미니쉬드 옥타토닉
            },
            "blues": {
                "primary": [0, 3, 5, 6, 7, 10],  # 블루스 스케일
                "secondary": [0, 2, 3, 5, 6, 7, 10]  # 마이너 블루스 펜타토닉
            }
        }
        
        return scale_chord_mapping
    
    def analyze_midi(self, midi_data: bytes) -> Dict:
        """
        MIDI 데이터 분석 및 음악적 특성 추출
        
        Args:
            midi_data: MIDI 파일 데이터 (바이트)
            
        Returns:
            음악적 특성이 포함된 컨텍스트 딕셔너리
        """
        # 음표 데이터 추출 (간소화된 구현)
        notes = self._extract_notes(midi_data)
        
        # 피치 클래스 분포 계산
        pitch_histogram = self._calculate_pitch_histogram(notes)
        
        # 키와 코드 진행 추론
        key = self._infer_key(pitch_histogram)
        mode = "major" if sum(pitch_histogram[0::2]) > sum(pitch_histogram[1::2]) else "minor"
        chord_progression = self._infer_chord_progression(notes, key, mode)
        
        # 리듬 특성 추출
        rhythm_features = self._analyze_rhythm(notes)
        
        # 결과 컨텍스트 생성
        context = {
            "midi_info": {
                "num_notes": len(notes),
                "duration": max([n["time"] + n["duration"] for n in notes]) if notes else 0
            },
            "harmonic_context": {
                "key": key,
                "mode": mode,
                "chord_progression": chord_progression
            },
            "melodic_features": {
                "pitch_histogram": pitch_histogram,
                "contour": self._analyze_contour(notes),
                "range": self._calculate_range(notes)
            },
            "rhythmic_features": rhythm_features
        }
        
        return context
    
    def _extract_notes(self, midi_data: bytes) -> List[Dict]:
        """MIDI 데이터에서 노트 정보 추출"""
        try:
            import io
            import mido
            
            notes = []
            midi_file = mido.MidiFile(file=io.BytesIO(midi_data))
            
            for track in midi_file.tracks:
                current_time = 0
                active_notes = {}
                
                for msg in track:
                    current_time += msg.time
                    
                    if msg.type == 'note_on' and msg.velocity > 0:
                        # 노트 시작
                        active_notes[msg.note] = {
                            "pitch": msg.note,
                            "time": current_time,
                            "velocity": msg.velocity
                        }
                    elif (msg.type == 'note_off' or 
                          (msg.type == 'note_on' and msg.velocity == 0)):
                        # 노트 종료
                        if msg.note in active_notes:
                            note_data = active_notes[msg.note]
                            note_data["duration"] = current_time - note_data["time"]
                            notes.append(note_data)
                            del active_notes[msg.note]
            
            return notes
            
        except Exception as e:
            print(f"MIDI 노트 추출 오류: {str(e)}")
            return []
    
    def _calculate_pitch_histogram(self, notes: List[Dict]) -> List[float]:
        """피치 클래스 분포 계산"""
        histogram = [0] * 12
        
        for note in notes:
            pitch_class = note["pitch"] % 12
            histogram[pitch_class] += 1
        
        # 정규화
        total = sum(histogram)
        if total > 0:
            histogram = [h / total for h in histogram]
        
        return histogram
    
    def _infer_key(self, pitch_histogram: List[float]) -> str:
        """피치 분포에서 키 추론"""
        key_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        weights = self.chord_analyzer["key_detection_weights"]
        
        # 메이저와 마이너 키 상관관계 계산
        major_corr = []
        minor_corr = []
        
        for i in range(12):  # 모든 가능한 키
            major_sum = 0
            minor_sum = 0
            
            for j in range(12):  # 각 피치 클래스
                major_sum += pitch_histogram[j] * weights[0][(j - i) % 12]
                minor_sum += pitch_histogram[j] * weights[1][(j - i) % 12]
            
            major_corr.append(major_sum)
            minor_corr.append(minor_sum)
        
        # 최대 상관관계 찾기
        max_major = max(major_corr)
        max_minor = max(minor_corr)
        
        if max_major > max_minor:
            # 메이저 키
            key_idx = major_corr.index(max_major)
            return key_names[key_idx]
        else:
            # 마이너 키
            key_idx = minor_corr.index(max_minor)
            return key_names[key_idx] + "m"
    
    def _infer_chord_progression(self, notes: List[Dict], key: str, mode: str) -> List[str]:
        """노트 데이터에서 코드 진행 추론"""
        # 단순화된 구현: 키에 기반한 일반적인 코드 진행 반환
        key_base = key[0] if len(key) == 1 else key[0:2]
        is_minor = key.endswith("m") or mode == "minor"
        
        note_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        key_idx = note_names.index(key_base)
        
        if is_minor:
            # 마이너 키 일반적 코드 진행: i - iv - v - i
            i = note_names[key_idx] + "m7"
            iv = note_names[(key_idx + 5) % 12] + "m7"
            v = note_names[(key_idx + 7) % 12] + "7"
            return [i, iv, v, i]
        else:
            # 메이저 키 일반적 코드 진행: I - vi - ii - V - I
            I = note_names[key_idx] + "maj7"
            vi = note_names[(key_idx + 9) % 12] + "m7"
            ii = note_names[(key_idx + 2) % 12] + "m7"
            V = note_names[(key_idx + 7) % 12] + "7"
            return [I, vi, ii, V, I]
    
    def _analyze_rhythm(self, notes: List[Dict]) -> Dict:
        """노트 데이터에서 리듬 특성 분석"""
        if not notes:
            return {
                "tempo": 120,
                "density": 0,
                "syncopation": 0
            }
        
        # 노트 밀도 계산
        duration = max([n["time"] + n["duration"] for n in notes])
        density = len(notes) / duration if duration > 0 else 0
        
        # 싱코페이션 감지 (오프비트에 있는 노트 비율)
        off_beat_count = 0
        for note in notes:
            # 4분음표 단위로 상대적 위치 계산
            relative_pos = (note["time"] * 4) % 1
            # 오프비트 감지
            if 0.2 < relative_pos < 0.3 or 0.45 < relative_pos < 0.55 or 0.7 < relative_pos < 0.8:
                off_beat_count += 1
        
        syncopation = off_beat_count / len(notes) if notes else 0
        
        return {
            "tempo": 120,  # 간단한 구현에서는 고정 값 사용
            "density": density,
            "syncopation": syncopation
        }
    
    def _analyze_contour(self, notes: List[Dict]) -> str:
        """멜로디 윤곽 분석"""
        if not notes:
            return "flat"
            
        # 시간순으로 정렬
        sorted_notes = sorted(notes, key=lambda n: n["time"])
        
        # 첫 번째와 마지막 음표의 피치 비교
        first_pitch = sorted_notes[0]["pitch"]
        last_pitch = sorted_notes[-1]["pitch"]
        
        if last_pitch > first_pitch + 3:
            return "ascending"
        elif last_pitch < first_pitch - 3:
            return "descending"
        else:
            # 피치 변화 추세 계산
            pitches = [n["pitch"] for n in sorted_notes]
            changes = [pitches[i+1] - pitches[i] for i in range(len(pitches)-1)]
            
            # 변화 방향 분석
            up_count = sum(1 for c in changes if c > 0)
            down_count = sum(1 for c in changes if c < 0)
            
            if up_count > down_count * 2:
                return "ascending"
            elif down_count > up_count * 2:
                return "descending"
            elif up_count > len(changes) * 0.4 and down_count > len(changes) * 0.4:
                return "undulating"
            else:
                return "arch"
    
    def _calculate_range(self, notes: List[Dict]) -> int:
        """음표 범위 계산 (반음 단위)"""
        if not notes:
            return 0
            
        pitches = [note["pitch"] for note in notes]
        return max(pitches) - min(pitches)
    
    def get_style_parameters(self, style: str) -> Dict:
        """스타일별 매개변수 가져오기"""
        if style in self.style_characteristics:
            return self.style_characteristics[style]
        else:
            # 기본 스타일 반환
            return self.style_characteristics["jazz_funk"]
    
    def get_scale_for_chord(self, chord: str, style: str) -> List[int]:
        """코드에 적합한 스케일 가져오기"""
        # 코드 파싱
        root = ""
        chord_type = ""
        
        # 코드 루트 추출
        for i, char in enumerate(chord):
            if char.isalpha() or char in ['#', 'b']:
                root += char
            else:
                chord_type = chord[i:]
                break
                
        if not chord_type:
            chord_type = "maj"  # 기본값
        
        # 코드 타입 정규화
        if chord_type == "m":
            chord_type = "min"
        elif chord_type == "m7":
            chord_type = "m7"
        elif chord_type == "M7":
            chord_type = "maj7"
        
        # 루트 음표 인덱스 찾기
        note_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        if root in note_names:
            root_idx = note_names.index(root)
        else:
            root_idx = 0  # 기본값 C
        
        # 코드에 적합한 스케일 선택
        if chord_type in self.scale_chord_mapping:
            # 스타일에 따라 primary 또는 secondary 스케일 선택
            if style == "blues":
                intervals = self.scale_chord_mapping["blues"]["primary"]
            elif style == "bebop" and ("7" in chord_type):
                # 비밥 스타일용 도미넌트 비밥 스케일
                intervals = [0, 2, 4, 5, 7, 9, 10, 11]
            else:
                intervals = self.scale_chord_mapping[chord_type]["primary"]
            
            # 루트 노트에 상대적인 스케일 노트 계산
            scale = [(root_idx + interval) % 12 for interval in intervals]
            return scale
        else:
            # 기본 메이저 스케일 반환
            intervals = self.scale_chord_mapping["maj"]["primary"]
            scale = [(root_idx + interval) % 12 for interval in intervals]
            return scale
    
    def save(self, path: str):
        """모델 저장"""
        model_data = {
            "metadata": self.metadata,
            "chord_analyzer": self.chord_analyzer,
            "style_characteristics": self.style_characteristics,
            "scale_chord_mapping": self.scale_chord_mapping
        }
        
        with open(path, "w") as f:
            json.dump(model_data, f, indent=2)
    
    @classmethod
    def load(cls, path: str):
        """모델 로드"""
        with open(path, "r") as f:
            model_data = json.load(f)
        
        model = cls()
        model.metadata = model_data["metadata"]
        model.chord_analyzer = model_data["chord_analyzer"]
        model.style_characteristics = model_data["style_characteristics"]
        model.scale_chord_mapping = model_data["scale_chord_mapping"]
        
        return model

# 모델 인스턴스 생성 및 저장 (첫 실행 시 사용)
if __name__ == "__main__":
    model = YueBaseModel()
    model.save("yue_base_model")
    print("YuE 기본 모델이 생성되고 저장되었습니다.")