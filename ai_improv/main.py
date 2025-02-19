from midi_rag import MIDIRAGSystem
import glob

def main():
    # RAG 시스템 초기화
    rag_system = MIDIRAGSystem()
    
    # 학습용 MIDI 파일들 로드
    training_midis = glob.glob("data/training/*.mid")
    rag_system.train(training_midis)
    
    # 새로운 MIDI 생성
    input_midi = "data/input/example.mid"
    output_midi = rag_system.generate(input_midi)
    
    # MIDI 파일 저장
    with open("data/output/generated.mid", "wb") as f:
        f.write(output_midi)

if __name__ == "__main__":
    main()