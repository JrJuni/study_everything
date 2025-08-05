import json
from deep_translator import GoogleTranslator # deep_translator 라이브러리가 설치되어 있어야 합니다.

# ---------------------------------------------------------------------------
# 이전에 우리가 함께 만들었던 일본어 -> 한국어 번역 함수
# (실제 코드에서는 이 함수를 별도 파일에 두고 import 하거나, 이 코드 상단에 정의해두세요)
def translate_japanese_to_korean(japanese_text):
    """
    일본어 텍스트를 한국어로 번역합니다.
    """
    if not japanese_text or not japanese_text.strip():
        # print("번역할 텍스트가 비어있어 건너뜁니다.") # 너무 자주 호출될 수 있으니 조용히 처리
        return "" # 빈 텍스트는 빈 텍스트로 반환
        
    try:
        translator = GoogleTranslator(source='ja', target='ko')
        translated_text = translator.translate(text=japanese_text)
        return translated_text
    except Exception as e:
        # 개별 세그먼트 번역 실패 시 오류를 출력하고, 원본 텍스트나 빈 문자열을 반환할 수 있습니다.
        print(f"경고: 텍스트 '{japanese_text[:20]}...' 번역 중 오류 - {e}")
        return None # 또는 japanese_text (원본 유지) 또는 "" (빈칸) 등으로 처리
# ---------------------------------------------------------------------------

def translate_segments_from_json(input_json_path, output_translated_json_path):
    """
    JSON 파일에서 일본어 세그먼트를 로드하여 한국어로 번역하고,
    번역된 세그먼트(원본 시간 정보 및 기타 정보 유지)를 새로운 JSON 파일로 저장합니다.

    :param input_json_path: 일본어 세그먼트 정보가 담긴 입력 JSON 파일 경로
    :param output_translated_json_path: 번역된 한국어 세그먼트 정보를 저장할 JSON 파일 경로
    :return: 번역된 세그먼트 리스트 또는 None (오류 발생 시)
    """
    try:
        with open(input_json_path, "r", encoding="utf-8") as f:
            japanese_segments = json.load(f)
        print(f"성공: '{input_json_path}'에서 {len(japanese_segments)}개의 일본어 세그먼트를 로드했습니다.")
    except FileNotFoundError:
        print(f"오류: 입력 JSON 파일 '{input_json_path}'을(를) 찾을 수 없습니다. 경로를 확인해주세요.")
        return None
    except json.JSONDecodeError:
        print(f"오류: 입력 JSON 파일 '{input_json_path}'이(가) 올바른 JSON 형식이 아닙니다.")
        return None
    except Exception as e:
        print(f"오류: JSON 파일 로드 중 예상치 못한 오류 발생 - {e}")
        return None

    translated_segments_data = []
    total_segments = len(japanese_segments)
    print(f"\n총 {total_segments}개의 세그먼트에 대한 한국어 번역을 시작합니다...")

    for index, segment in enumerate(japanese_segments):
        original_ja_text = segment.get("text", "") # 세그먼트에서 일본어 텍스트 추출
        
        # 번역 수행
        korean_text = translate_japanese_to_korean(original_ja_text)
        
        if korean_text is None:
            # 번역 실패 시, 로그를 남기고 빈 텍스트로 대체하거나 원본을 유지할 수 있습니다.
            # 여기서는 빈 텍스트로 처리합니다.
            print(f"  세그먼트 {index + 1}/{total_segments}의 번역 실패. 빈 텍스트로 처리합니다.")
            korean_text = "" 

        # 원본 세그먼트의 모든 정보를 복사하고, 'text' 필드만 번역된 한국어로 업데이트
        # 이렇게 하면 'start', 'end' 시간 및 Whisper가 제공한 다른 모든 메타데이터가 유지됩니다.
        translated_segment_info = segment.copy()
        translated_segment_info["text"] = korean_text
        # 필요하다면 원본 일본어 텍스트도 함께 저장할 수 있습니다.
        # translated_segment_info["original_japanese_text"] = original_ja_text
        
        translated_segments_data.append(translated_segment_info)

        # 진행 상황 표시 (10개 세그먼트마다 또는 마지막 세그먼트에서)
        if (index + 1) % 10 == 0 or (index + 1) == total_segments:
            print(f"  진행: {index + 1}/{total_segments} 세그먼트 번역 완료...")

    # 번역된 세그먼트 데이터를 새로운 JSON 파일로 저장
    try:
        with open(output_translated_json_path, "w", encoding="utf-8") as f:
            json.dump(translated_segments_data, f, ensure_ascii=False, indent=4)
        print(f"\n성공: 번역된 세그먼트 정보가 '{output_translated_json_path}' 파일로 저장되었습니다.")
    except IOError as e:
        print(f"오류: 번역된 JSON 파일 저장 중 오류 발생 - {e}")
        # 파일 저장에 실패해도 데이터는 반환
    
    return translated_segments_data

# --- 사용 예시 ---
# 1. Whisper를 통해 추출되어 JSON으로 저장된 일본어 세그먼트 파일
input_japanese_segments_json = "C:/Users/Junyeob/Downloads/japanese_audio_segments.json" 

# 2. 번역된 한국어 세그먼트 정보를 저장할 새 JSON 파일 이름
output_korean_segments_json = "C:/Users/Junyeob/Downloads/korean_translated_segments.json"

print("="*50)
print(f"입력 파일: {input_japanese_segments_json}")
print(f"출력 파일 (번역된 내용): {output_korean_segments_json}")
print("="*50)

# 함수 실행
korean_segments_result = translate_segments_from_json(input_japanese_segments_json, output_korean_segments_json)

if korean_segments_result:
    print(f"\n총 {len(korean_segments_result)}개의 세그먼트가 성공적으로 번역(또는 처리)되어 저장되었습니다.")
    # 첫 번째 번역된 세그먼트 데이터 예시 (JSON 파일에도 동일한 구조로 저장됨)
    if len(korean_segments_result) > 0:
        print("\n--- 첫 번째 번역된 세그먼트 데이터 (예시) ---")
        print(f"  시작 시간: {korean_segments_result[0]['start']}")
        print(f"  종료 시간: {korean_segments_result[0]['end']}")
        print(f"  번역된 텍스트 (한국어): {korean_segments_result[0]['text']}")
        # 만약 'original_japanese_text'도 함께 저장했다면 여기서 출력해볼 수 있습니다.
        print("---------------------------------------------")
else:
    print("\n번역 작업에 실패했거나 처리된 세그먼트가 없습니다. 오류 메시지를 확인해주세요.")