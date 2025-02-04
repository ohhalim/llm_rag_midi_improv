# from midi_input import get_midi_input
# from llm_api import query_llm_rag
# from midi_output import play_midi_output

# if __name__ == "__main__":
#     while True:
#         midi_data = get_midi_input()
#         llm_response = query_llm_rag(midi_data)
#         play_midi_output(llm_response)
    


from midi_input import get_midi_input
from llm_api import query_llm_rag, generate_fake_midi_data
from midi_output import play_midi_output

def test_llm_rag():
    # 테스트 실행
    fake_midi_input = generate_fake_midi_data()
    print("입력 MIDI 데이터:", fake_midi_input)

    llm_response = query_llm_rag(fake_midi_input)
    print("LLM RAG 응답:", llm_response)
    print("응답 길이:", len(llm_response))

if __name__ == "__main__":
    # 테스트 모드
    test_llm_rag()

    # 실제 실행 모드 (주석 처리)
    # while True:
    #     midi_data = get_midi_input()
    #     llm_response = query_llm_rag(midi_data)
    #     play_midi_output(llm_response)
