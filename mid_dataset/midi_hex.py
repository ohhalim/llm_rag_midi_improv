# MIDI 파일 열기
with open('moaninfunk.mid', 'rb') as f:
    data = f.read()

# 16진수로 출력
print(data.hex())
