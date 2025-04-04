"""
MIDI RAG 시스템 메인 실행 파일
시스템을 실행하고 관리하는 명령줄 인터페이스를 제공합니다.
"""
import argparse
import asyncio
import os
import glob
from pathlib import Path

from midi_rag_system.core.midi_rag_system import MIDIRAGSystem
from midi_rag_system.api.server import start_server
from midi_rag_system.core.mcp_server import MIDIMCPServer
import config

def parse_args():
    """명령줄 인수 파싱"""
    parser = argparse.ArgumentParser(description="MIDI RAG 시스템")
    
    # 서브 커맨드 설정
    subparsers = parser.add_subparsers(dest="command", help="명령")
    
    # 서버 실행 커맨드
    server_parser = subparsers.add_parser("server", help="API 서버 실행")
    server_parser.add_argument("--host", type=str, default=config.API_HOST, help="서버 호스트")
    server_parser.add_argument("--port", type=int, default=config.API_PORT, help="서버 포트")
    server_parser.add_argument("--debug", action="store_true", help="디버그 모드 활성화")
    
    # MCP 서버 실행 커맨드
    mcp_parser = subparsers.add_parser("mcp", help="MCP 서버 실행")
    mcp_parser.add_argument("--host", type=str, default=config.MCP_SERVER_HOST, help="MCP 서버 호스트")
    mcp_parser.add_argument("--port", type=int, default=config.MCP_SERVER_PORT, help="MCP 서버 포트")
    
    # 학습 커맨드
    train_parser = subparsers.add_parser("train", help="MIDI 파일로 시스템 학습")
    train_parser.add_argument("--dir", type=str, default=str(config.TRAINING_DIR), help="학습 디렉토리")
    train_parser.add_argument("--save", type=str, default=config.VECTORSTORE_PATH, help="벡터 저장소 저장 경로")
    
    # 생성 커맨드
    generate_parser = subparsers.add_parser("generate", help="MIDI 생성")
    generate_parser.add_argument("input", type=str, help="입력 MIDI 파일 경로")
    generate_parser.add_argument("--output", type=str, help="출력 파일 경로")
    generate_parser.add_argument("--format", type=str, choices=["midi", "json"], default="midi", help="출력 형식")
    generate_parser.add_argument("--genre", type=str, help="장르 및 스타일 설명")
    generate_parser.add_argument("--lyrics", type=str, help="가사 또는 설명")
    
    return parser.parse_args()

async def generate_midi(args):
    """MIDI 생성 함수"""
    # MIDI RAG 시스템 초기화
    rag_system = MIDIRAGSystem()
    
    # 벡터 저장소 로드
    if os.path.exists(config.VECTORSTORE_PATH):
        print(f"벡터 저장소 로드 중: {config.VECTORSTORE_PATH}")
        rag_system.load_vectorstore(config.VECTORSTORE_PATH)
    else:
        print("경고: 벡터 저장소를 찾을 수 없습니다. 먼저 'train' 명령을 실행하세요.")
    
    # 입력 파일 확인
    if not os.path.exists(args.input):
        print(f"오류: 입력 파일을 찾을 수 없습니다: {args.input}")
        return
    
    # 출력 파일 경로 설정
    if args.output:
        output_path = args.output
    else:
        input_name = os.path.basename(args.input)
        output_name = f"generated_{os.path.splitext(input_name)[0]}.{args.format}"
        output_path = os.path.join(str(config.OUTPUT_DIR), output_name)
    
    # 장르 및 가사 설정
    genre_text = args.genre or "장르: 팝, 스타일: 현대적, 악기: 피아노, 기타, 드럼"
    lyrics_text = args.lyrics or "즉흥 연주를 위한 멜로디"
    
    print(f"입력 MIDI: {args.input}")
    print(f"출력 형식: {args.format}")
    print(f"출력 경로: {output_path}")
    print("MIDI 생성 중...")
    
    # MIDI 생성
    midi_data = await rag_system.generate(
        args.input,
        output_format=args.format,
        genre_text=genre_text,
        lyrics_text=lyrics_text
    )
    
    # MIDI 저장
    rag_system.save_midi(midi_data, output_path)
    print(f"MIDI 생성 완료: {output_path}")

def train_system(args):
    """시스템 학습 함수"""
    # MIDI RAG 시스템 초기화
    rag_system = MIDIRAGSystem()
    
    # 학습 디렉토리에서 MIDI 파일 목록 가져오기
    midi_files = glob.glob(os.path.join(args.dir, "*.mid")) + glob.glob(os.path.join(args.dir, "*.midi"))
    
    if not midi_files:
        print(f"오류: 학습 디렉토리에 MIDI 파일이 없습니다: {args.dir}")
        return
    
    print(f"학습 파일 수: {len(midi_files)}")
    print("학습 중...")
    
    # 학습 실행
    rag_system.train(midi_files, args.save)
    print(f"학습 완료. 벡터 저장소가 저장되었습니다: {args.save}")

def run_server(args):
    """API 서버 실행 함수"""
    # 환경 변수 설정
    os.environ["API_HOST"] = args.host
    os.environ["API_PORT"] = str(args.port)
    os.environ["API_DEBUG"] = str(args.debug).lower()
    
    print(f"API 서버 시작: {args.host}:{args.port}")
    start_server()

def run_mcp_server(args):
    """MCP 서버 실행 함수"""
    # 환경 변수 설정
    os.environ["MCP_SERVER_HOST"] = args.host
    os.environ["MCP_SERVER_PORT"] = str(args.port)
    
    print(f"MCP 서버 시작: {args.host}:{args.port}")
    mcp_server = MIDIMCPServer(host=args.host, port=args.port)
    mcp_server.start()

def main():
    """메인 함수"""
    args = parse_args()
    
    if args.command == "server":
        run_server(args)
    elif args.command == "mcp":
        run_mcp_server(args)
    elif args.command == "train":
        train_system(args)
    elif args.command == "generate":
        asyncio.run(generate_midi(args))
    else:
        print("명령을 지정하세요. 도움말을 보려면 --help를 사용하세요.")

if __name__ == "__main__":
    main()
