import whisper
import datetime # 시간 포맷팅을 위해 import
import json # 혹시 중간 결과(영어 세그먼트)를 json으로도 저장하고 싶다면 사용

# 초 단위를 SRT 시간 형식(HH:MM:SS,ms)으로 바꿔주는 함수
def format_timestamp_srt(seconds_float):
    if seconds_float < 0:
        seconds_float = 0
    delta = datetime.timedelta(seconds=seconds_float)
    hours, remainder = divmod(delta.seconds, 3600)
    minutes, seconds_int = divmod(remainder, 60)
    milliseconds = delta.microseconds // 1000
    return f"{hours:02}:{minutes:02}:{seconds_int:02},{milliseconds:03}"

def japanese_audio_to_english_srt(audio_path, output_srt_path, model_size="large-v3"):
    """
    일본어 음성 파일을 영어 텍스트로 변환하고, 시간 정보를 포함하여 영어 SRT 자막 파일을 생성합니다.

    :param audio_path: 일본어 음성 파일 경로
    :param output_srt_path: 생성될 영어 SRT 파일 경로
    :param model_size: 사용할 Whisper 모델 크기
    :return: 성공 시 True, 실패 시 False
    """
    try:
        print(f"Whisper 모델({model_size}) 로드 중...")
        model = whisper.load_model(model_size)
        print("모델 로드 완료.")

        print(f"'{audio_path}' 파일의 일본어 음성을 영어 텍스트로 직접 변환 및 타임스탬프 추출 중...")
        # language="ja"로 지정하여 원본 오디오가 일본어임을 알리고,
        # task="translate"로 지정하여 영어로 번역된 텍스트를 요청합니다.
        result = model.transcribe(audio_path, language="ja", task="translate", verbose=True)
        
        english_segments = result.get("segments")

        if not english_segments:
            print("오류: 음성에서 세그먼트(영어 번역 포함)를 추출하지 못했습니다.")
            return False

        print(f"총 {len(english_segments)}개의 영어 텍스트 세그먼트를 추출했습니다.")
        
        # (선택 사항) 추출된 영어 세그먼트를 JSON 파일로도 저장하고 싶다면:
        # output_english_segments_json = output_srt_path.replace(".srt", "_segments.json")
        # try:
        #     with open(output_english_segments_json, "w", encoding="utf-8") as f_json:
        #         json.dump(english_segments, f_json, ensure_ascii=False, indent=4)
        #     print(f"영어 세그먼트 정보가 '{output_english_segments_json}' 파일로 저장되었습니다.")
        # except IOError as e_json:
        #     print(f"JSON 파일 저장 중 오류: {e_json}")

        # SRT 파일 생성 시작
        srt_content = []
        valid_segment_counter = 0
        for i, segment_data in enumerate(english_segments):
            start_time_sec = segment_data.get("start")
            end_time_sec = segment_data.get("end")
            # 이제 "text" 키에는 영어로 번역된 내용이 들어있습니다.
            english_text = segment_data.get("text", "").strip()

            if start_time_sec is None or end_time_sec is None or not english_text:
                print(f"  정보: {i+1}번째 세그먼트는 시간 정보가 없거나 텍스트가 비어있어 건너뜁니다.")
                continue
            
            valid_segment_counter += 1
            start_time_str = format_timestamp_srt(float(start_time_sec))
            end_time_str = format_timestamp_srt(float(end_time_sec))
            
            srt_content.append(str(valid_segment_counter))
            srt_content.append(f"{start_time_str} --> {end_time_str}")
            srt_content.append(english_text)
            srt_content.append("")

        with open(output_srt_path, "w", encoding="utf-8") as f_srt:
            f_srt.write("\n".join(srt_content))
        
        print(f"\n성공: 영어 SRT 자막 파일이 '{output_srt_path}' 경로에 생성되었습니다.")
        return True
            
    except Exception as e:
        print(f"SRT 파일 생성 중 전체 오류 발생: {e}")
        return False

# --- 사용 예시 ---
# 1. 일본어 음성 파일 경로 (예: "extracted_japanese_audio.mp3" 또는 원본 mp3 파일)
japanese_audio_input_file = "C:/Users/Junyeob/Downloads/video_test01.mp3" 

# 2. 최종적으로 생성될 영어 SRT 자막 파일 이름
output_english_srt_file = "C:/Users/Junyeob/Downloads/video_test01.srt"

print("="*50)
print(f"입력 일본어 오디오 파일: {japanese_audio_input_file}")
print(f"출력 영어 SRT 파일: {output_english_srt_file}")
print("="*50)

# (주의: 이 코드를 실행하면 Whisper 음성 인식 및 영어 번역을 다시 수행합니다!)
if japanese_audio_to_english_srt(japanese_audio_input_file, output_english_srt_file, model_size="large-v3"):
    print(f"\n'{output_english_srt_file}' 영어 자막 파일 생성을 완료했습니다!")
    print("영상 플레이어에서 자막을 불러와 확인해보세요! 😊")
else:
    print("\n영어 자막 파일 생성에 실패했습니다. 오류 메시지를 확인해주세요.")