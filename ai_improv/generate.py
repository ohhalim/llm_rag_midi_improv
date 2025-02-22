import glob
from midi_rag import MIDIRAGSystem
# python generate.py


# MIDI RAG 시스템 초기화
rag_system = MIDIRAGSystem()

# 모든 트레이닝 파일 로드
training_files = glob.glob('data/training/*.mid')
print(f"학습에 사용할 MIDI 파일 수: {len(training_files)}")

# 시스템 학습
rag_system.train(training_files)

# 입력 MIDI 선택
input_midi = 'data/training/corazon_Jacob_Collier.mid'  # 테스트에 사용한 동일한 파일

# MIDI 생성 (JSON 형식)
json_output = rag_system.generate(input_midi, output_format='json')
rag_system.save_midi(json_output, 'data/output/output.json')

# MIDI 생성 (MIDI 형식)
midi_output = rag_system.generate(input_midi, output_format='midi')
rag_system.save_midi(midi_output, 'data/output/output.mid')

print("MIDI 생성 완료!")

# 잘되는거 맞아?
# 지금 까지 진행상황 뭐 어느정도 나오는거 같은데 미디파일 생성이 안돼 