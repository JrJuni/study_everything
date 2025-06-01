import json
import datetime # 시간 포맷 변환을 위해 필요합니다.

# 이전에 사용했던, 초 시간을 SRT 형식 문자열로 바꿔주는 함수입니다.
def format_timestamp_srt(seconds_float):
    """초 단위 시간(float)을 SRT 자막 형식(HH:MM:SS,ms) 문자열로 변환합니다."""
    # timedelta는 음수 초를 처리하지 못하므로, 0 미만이면 0으로 처리 (혹시 모를 오류 방지)
    if seconds_float < 0:
        seconds_float = 0
    delta = datetime.timedelta(seconds=seconds_float)
    hours, remainder = divmod(delta.seconds, 3600)
    minutes, seconds_int = divmod(remainder, 60) # 변수 이름 충돌 방지를 위해 seconds_int로 변경
    milliseconds = delta.microseconds // 1000 # 마이크로초를 밀리초로 변환
    return f"{hours:02}:{minutes:02}:{seconds_int:02},{milliseconds:03}"

def generate_srt_from_korean_json(korean_segments_json_path, output_srt_file_path):
    """
    번역된 한국어 세그먼트 정보가 담긴 JSON 파일을 읽어 SRT 자막 파일을 생성합니다.

    :param korean_segments_json_path: 한국어 텍스트와 시간 정보가 담긴 JSON 파일 경로
    :param output_srt_file_path: 생성될 SRT 파일 경로
    :return: 성공 시 True, 실패 시 False
    """
    try:
        with open(korean_segments_json_path, "r", encoding="utf-8") as f:
            korean_segments = json.load(f)
        print(f"성공: '{korean_segments_json_path}'에서 {len(korean_segments)}개의 한국어 세그먼트를 로드했습니다.")
    except FileNotFoundError:
        print(f"오류: 입력 JSON 파일 '{korean_segments_json_path}'을(를) 찾을 수 없습니다. 경로를 확인해주세요.")
        return False
    except json.JSONDecodeError:
        print(f"오류: 입력 JSON 파일 '{korean_segments_json_path}'이(가) 올바른 JSON 형식이 아닙니다.")
        return False
    except Exception as e:
        print(f"오류: JSON 파일 로드 중 예상치 못한 오류 발생 - {e}")
        return False

    srt_content = []
    segment_counter = 0

    for i, segment_data in enumerate(korean_segments):
        # JSON 파일에서 'start', 'end', 'text' 키를 사용합니다.
        # 이 키 이름은 이전 단계(번역된 JSON 저장 시)에서 사용한 키와 일치해야 합니다.
        start_time_sec = segment_data.get("start")
        end_time_sec = segment_data.get("end")
        korean_text = segment_data.get("text", "").strip() # 앞뒤 공백 제거

        # 시간 정보가 없거나, 텍스트가 비어있으면 해당 세그먼트는 건너뛸 수 있습니다.
        if start_time_sec is None or end_time_sec is None:
            print(f"  경고: {i+1}번째 세그먼트에 시간 정보가 없어 건너뜁니다. (텍스트: '{korean_text[:30]}...')")
            continue
        
        if not korean_text: # 번역된 텍스트가 비어있다면 자막으로 만들지 않음
            print(f"  정보: {i+1}번째 세그먼트의 텍스트 내용이 비어있어 자막으로 생성하지 않습니다.")
            continue

        segment_counter += 1 # 유효한 자막 세그먼트 카운터

        # 시간 포맷 변환
        start_time_str = format_timestamp_srt(float(start_time_sec))
        end_time_str = format_timestamp_srt(float(end_time_sec))
        
        # SRT 형식으로 문자열 만들기
        srt_content.append(str(segment_counter)) # 자막 번호
        srt_content.append(f"{start_time_str} --> {end_time_str}") # 시간
        srt_content.append(korean_text) # 번역된 한국어 자막 내용
        srt_content.append("") # 빈 줄로 각 자막 구분

    # 만들어진 SRT 내용을 파일에 쓰기
    try:
        with open(output_srt_file_path, "w", encoding="utf-8") as f:
            f.write("\n".join(srt_content))
        print(f"\n성공: SRT 자막 파일이 '{output_srt_file_path}' 경로에 생성되었습니다.")
        return True
    except IOError as e:
        print(f"오류: SRT 파일 저장 중 오류 발생 - {e}")
        return False
    except Exception as e:
        print(f"오류: SRT 파일 생성 중 예상치 못한 오류 발생 - {e}")
        return False

# --- 사용 예시 ---
# 1. 이전 단계에서 생성한, 한국어 텍스트와 원본 시간 정보가 담긴 JSON 파일
input_korean_segments_json = "C:/Users/Junyeob/Downloads/korean_translated_segments.json" 

# 2. 최종적으로 생성될 한국어 SRT 자막 파일 이름
output_srt_file = "C:/Users/Junyeob/Downloads/video_test01.srt"

print("="*50)
print(f"입력 파일 (번역된 한국어 세그먼트 JSON): {input_korean_segments_json}")
print(f"출력 파일 (SRT 자막): {output_srt_file}")
print("="*50)

# 함수 실행
if generate_srt_from_korean_json(input_korean_segments_json, output_srt_file):
    print(f"\n'{output_srt_file}' 한국어 자막 파일 생성을 완료했습니다!")
    print("이제 이 SRT 파일을 영상 플레이어에서 자막으로 불러와 확인해보세요! 😊")
else:
    print("\nSRT 자막 파일 생성에 실패했습니다. 오류 메시지를 확인해주세요.")