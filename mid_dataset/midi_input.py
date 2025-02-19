import mido

# MIDI 입력 포트 확인
print(mido.get_input_names())

# MIDI 입력 읽기
with mido.open_input() as inport:
    for msg in inport:
        if msg.type in ['note_on', 'note_off']:
            print(f"입력된 메시지: {msg}")
