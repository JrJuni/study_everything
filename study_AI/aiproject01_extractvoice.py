from moviepy.editor import VideoFileClip

def extract_audio(video_path, audio_path):
    """
    영상 파일에서 음성을 추출하여 저장합니다.pip

    :param video_path: 원본 영상 파일 경로 (예: "my_video.mp4")
    :param audio_path: 저장할 음성 파일 경로 (예: "output_audio.mp3")
    """
    try:
        print(f"영상 파일 '{video_path}'에서 음성 추출을 시작합니다...")
        video_clip = VideoFileClip(video_path)
        audio_clip = video_clip.audio
        audio_clip.write_audiofile(audio_path)
        video_clip.close() # VideoFileClip 객체를 닫아줍니다.
        audio_clip.close() # AudioClip 객체를 닫아줍니다.
        print(f"음성 추출 완료! '{audio_path}' 파일로 저장되었습니다. 🎉")
        return True
    except Exception as e:
        print(f"음성 추출 중 오류가 발생했습니다: {e}")
        return False

# --- 아래 부분을 수정해서 사용하세요 ----
# 가지고 있는 일본어 영상 파일 경로를 넣어주세요.
# 예를 들어, "C:/Users/YourName/Videos/japanese_video.mp4" 와 같이요.
# 만약 파이썬 스크립트와 같은 폴더에 영상이 있다면 파일 이름만 적어도 됩니다.
input_video_file = "C:/Users/Junyeob/Downloads/video_test01.mp4"
print(f"DEBUG: input_video_file의 현재 값은 '{input_video_file}' 입니다.")

# 저장될 음성 파일의 이름과 경로를 정해주세요.
output_audio_file = "C:/Users/Junyeob/Downloads/video_test01.mp3" # 또는 .wav 도 좋아요!

# 함수 실행
if input_video_file != "영상경로_미수정.mp4":
    extract_audio(input_video_file, output_audio_file)
else:
    print("Error: 영상 파일 경로를 'input_video_file' 변수에 설정해주세요.")
    print("예시: input_video_file = 'my_japanese_video.mp4'")