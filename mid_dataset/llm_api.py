# import requests

# def query_llm_rag(midi_data):
#     url = "http://localhost:8000/api/midi"  # LLM API 주소
#     payload = {"midi_input": midi_data}
#     headers = {"Content-Type": "application/json"}

#     response = requests.post(url, json=payload, headers=headers)
#     if response.status_code == 200:
#         return response.json().get("midi_output")
#     else:
#         return [0x90, 0x40, 0x64, 0x80, 0x40, 0x40]  # 기본 응답 (에러 시)


import random

def generate_fake_midi_data(num_notes=5):
    fake_midi_data = []
    for _ in range(num_notes):
        note = random.randint(0x00, 0x7F)
        velocity = random.randint(0x00, 0x7F)
        note_on = [0x90, note, velocity]
        note_off = [0x80, note, 0x40]
        fake_midi_data.extend(note_on)
        fake_midi_data.extend(note_off)
    return fake_midi_data

def query_llm_rag(midi_data):
    if not midi_data or len(midi_data) < 6:
        return [0x90, 0x40, 0x64, 0x80, 0x40, 0x40]
    
    output_midi = []
    for i in range(0, len(midi_data), 3):
        if i + 2 < len(midi_data):
            note_on = [midi_data[i], 
                       min(127, midi_data[i+1] + random.randint(-5, 5)),
                       min(127, midi_data[i+2] + random.randint(-10, 10))]
            output_midi.extend(note_on)
    
    return output_midi
