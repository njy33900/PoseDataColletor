import tkinter as tk
from tkinter import messagebox
from tkinter import font
import sys

# 모듈 임포트
from collector_logic import DataCollectorLogic
from collector_gui import CollectorGUI
import collector_api

# ==========================================
# 설정
# ==========================================
CAMERA_SOURCE = "http://100.111.11.35:8080/video"


# ==========================================
# 실행 모드 함수
# ==========================================
def run_desktop_gui():
    """데스크탑 GUI 모드 실행"""
    try:
        print("🖥️ 데스크탑 GUI 모드로 시작합니다...")
        root = tk.Tk()
        # 로직 초기화
        logic = DataCollectorLogic(model_path='yolo11n-pose.pt', camera_source=CAMERA_SOURCE)
        # GUI 연결
        app = CollectorGUI(root, logic)
        # 종료 이벤트 연결
        root.protocol("WM_DELETE_WINDOW", app.on_close)
        root.mainloop()
    except Exception as e:
        print(f"오류 발생: {e}")


def run_web_server():
    """웹 서버 모드 실행"""
    try:
        print("🌐 웹 서버 모드로 시작합니다...")
        print("브라우저에서 접속하세요: http://localhost:8080/collector")
        # API 서버 실행 (Blocking)
        collector_api.run_api(camera_source=CAMERA_SOURCE)
    except Exception as e:
        print(f"오류 발생: {e}")


# ==========================================
# 런처 (모드 선택 창)
# ==========================================
def show_launcher():
    launcher = tk.Tk()
    launcher.title("행동 데이터 수집기")
    launcher.geometry("400x300")

    # 화면 중앙 배치
    screen_width = launcher.winfo_screenwidth()
    screen_height = launcher.winfo_screenheight()
    x_cordinate = int((screen_width / 2) - (400 / 2))
    y_cordinate = int((screen_height / 2) - (300 / 2))
    launcher.geometry(f"400x300+{x_cordinate}+{y_cordinate}")

    # 스타일 설정
    title_font = font.Font(family="Arial", size=16, weight="bold")
    btn_font = font.Font(family="Arial", size=12)

    # 선택된 모드 저장 변수
    selected_mode = [None]

    def on_gui_click():
        selected_mode[0] = "GUI"
        launcher.destroy()

    def on_web_click():
        selected_mode[0] = "WEB"
        launcher.destroy()

    # UI 구성
    tk.Label(launcher, text="실행 모드를 선택하세요", font=title_font, pady=20).pack()

    # 1. 데스크탑 GUI 버튼
    btn_gui = tk.Button(launcher, text="🖥️ 데스크탑 GUI 실행\n",
                        font=btn_font, bg="#e1f5fe", fg="black", width=25, height=3,
                        command=on_gui_click)
    btn_gui.pack(pady=10)

    # 2. 웹 서버 버튼
    btn_web = tk.Button(launcher, text="🌐 웹 서버 모드 실행\n",
                        font=btn_font, bg="#e8f5e9", fg="black", width=25, height=3,
                        command=on_web_click)
    btn_web.pack(pady=10)

    # 실행 및 대기
    launcher.mainloop()

    return selected_mode[0]


# ==========================================
# 메인 진입점
# ==========================================
if __name__ == "__main__":
    # 1. 런처 실행 (사용자 선택 대기)
    mode = show_launcher()

    # 2. 선택된 모드에 따라 실행
    if mode == "GUI":
        run_desktop_gui()
    elif mode == "WEB":
        run_web_server()
    else:
        print("프로그램을 종료합니다.")