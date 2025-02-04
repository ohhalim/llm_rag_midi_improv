import mido

def play_midi_output(midi_data):
    with mido.open_output() as outport:
        msg = mido.Message.from_bytes(midi_data)
        outport.send(msg)
