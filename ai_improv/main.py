#!/usr/bin/env python
"""
MIDI 프로세서 메인 스크립트
MIDI 파일 처리 예제
"""
import os
import sys
import argparse
import json
from midi_processor.core.processor import MIDIProcessor

def main():
    # 명령줄 인자 파싱
    parser = argparse.ArgumentParser(description='MIDI 파일 처리 도구')
    parser.add_argument('--input', '-i', dest='input_file', required=True,
                        help='처리할 MIDI 파일 경로')
    parser.add_argument('--output', '-o', dest='output_file',
                        help='처리 결과를 저장할 MIDI 파일 경로')
    parser.add_argument('--json', '-j', dest='json_file',
                        help='JSON 형식으로 저장할 파일 경로')
    parser.add_argument('--info', '-n', action='store_true',
                        help='MIDI 파일 정보만 출력')
    
    args = parser.parse_args()
    
    # 입력 파일 확인
    if not os.path.exists(args.input_file):
        print(f"오류: 입력 파일을 찾을 수 없습니다: {args.input_file}")
        return 1
    
    # MIDI 프로세서 초기화
    processor = MIDIProcessor()
    
    # MIDI 파일 로드
    if not processor.load(args.input_file):
        print(f"오류: MIDI 파일을 로드하지 못했습니다: {args.input_file}")
        return 1
    
    # MIDI 파일 정보 출력
    info = processor.get_info()
    print(f"MIDI 파일 정보:")
    print(f"- 틱/비트: {info['ticks_per_beat']}")
    print(f"- 트랙 수: {info['track_count']}")
    for track in info['tracks']:
        print(f"  - 트랙 {track['index']}: {track['name']} ({track['note_count']} 노트)")
    
    # 정보만 요청된 경우 종료
    if args.info:
        return 0
    
    # JSON 형식으로 변환하여 저장
    if args.json_file:
        json_data = processor.to_json()
        os.makedirs(os.path.dirname(os.path.abspath(args.json_file)), exist_ok=True)
        with open(args.json_file, 'w', encoding='utf-8') as f:
            f.write(json_data)
        print(f"MIDI 파일을 JSON으로 변환하여 저장했습니다: {args.json_file}")
    
    # 출력 파일로 저장
    if args.output_file:
        if processor.save(args.output_file):
            print(f"MIDI 파일을 저장했습니다: {args.output_file}")
        else:
            print(f"오류: MIDI 파일을 저장하지 못했습니다: {args.output_file}")
            return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 