import whisper
import json

def transcribe_and_save_segments_as_json(audio_path, output_json_path, model_size="large-v3", language="ja"):
    """
    음성 파일에서 텍스트 세그먼트와 시간 정보를 추출하여 JSON 파일로 저장합니다.

    :param audio_path: 음성 파일 경로
    :param output_json_path: 저장할 JSON 파일 경로
    :param model_size: 사용할 Whisper 모델 크기
    :param language: 음성 언어
    :return: 추출된 세그먼트 리스트 또는 None (오류 발생 시)
    """
    try:
        print(f"Whisper 모델({model_size})을 로드하는 중입니다...")
        model = whisper.load_model(model_size)
        print("모델 로드 완료!")

        print(f"'{audio_path}' 파일의 음성을 '{language}' 언어 텍스트 및 시간 정보로 변환 시작...")
        # verbose=True로 설정하면 처리 과정을 더 자세히 볼 수 있습니다.
        result = model.transcribe(audio_path, language=language, task="transcribe", verbose=True)
        
        segments = result.get("segments") # .get()을 사용하면 "segments" 키가 없을 경우 None을 반환

        if segments:
            print(f"총 {len(segments)}개의 음성 세그먼트를 추출했습니다.")
            
            # 추출된 세그먼트 정보를 JSON 파일로 저장
            try:
                with open(output_json_path, "w", encoding="utf-8") as f:
                    # ensure_ascii=False: 비 ASCII 문자(한글, 일본어 등)를 그대로 저장
                    # indent=4: JSON 파일을 사람이 보기 좋게 4칸 들여쓰기함
                    json.dump(segments, f, ensure_ascii=False, indent=4)
                print(f"세그먼트 정보가 '{output_json_path}' 파일로 성공적으로 저장되었습니다.")
            except IOError as e:
                print(f"JSON 파일 저장 중 오류 발생: {e}")
                # 파일 저장에 실패하더라도, 추출된 세그먼트 데이터는 반환합니다.
            
            return segments
        else:
            print("음성에서 세그먼트를 추출하지 못했습니다.")
            return None
            
    except Exception as e:
        print(f"음성 변환 또는 세그먼트 추출 중 오류 발생: {e}")
        return None

# --- 사용 예시 ---
# 입력 오디오 파일 경로 (이전에 추출한 음성 파일)
extracted_audio_file_path = "C:/Users/Junyeob/Downloads/extracted_japanese_audio.mp3" 

# 추출된 세그먼트 정보를 저장할 JSON 파일 경로와 이름
output_json_file = "japanese_audio_segments.json" 

print(f"'{extracted_audio_file_path}'의 세그먼트 추출 및 JSON 저장을 시작합니다.")
print(f"결과는 '{output_json_file}' 파일에 저장됩니다.")

# 함수 실행
# (주의: 이 코드를 실행하면 Whisper 음성 인식을 다시 수행합니다!)
japanese_segments_data = transcribe_and_save_segments_as_json(
    extracted_audio_file_path,
    output_json_file,
    model_size="large-v3", # 필요에 따라 모델 크기 변경
    language="ja"
)

if japanese_segments_data:
    print(f"\n작업 완료! 총 {len(japanese_segments_data)}개의 세그먼트가 처리되었습니다.")
    # 첫 번째 세그먼트 데이터 예시 출력 (JSON 파일에도 동일한 구조로 저장됨)
    if len(japanese_segments_data) > 0:
        print("\n--- 첫 번째 세그먼트 데이터 예시 ---")
        print(f"  시작 시간: {japanese_segments_data[0]['start']}")
        print(f"  종료 시간: {japanese_segments_data[0]['end']}")
        print(f"  텍스트: {japanese_segments_data[0]['text']}")
        print("-----------------------------------")
else:
    print("\n작업에 실패했거나 처리된 세그먼트가 없습니다.")