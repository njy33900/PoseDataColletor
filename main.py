import tkinter as tk
from collector_logic import DataCollectorLogic
from collector_gui import CollectorGUI


def main():
    # 메인 윈도우 생성
    root = tk.Tk()

    # 로직(Backend) 초기화
    # 카메라 소스: 0(웹캠) 또는 IP주소
    # camera_source = "http://100.75.20.5:8080/video"
    camera_source = "http://100.111.11.35:8080/video"
    logic = DataCollectorLogic(model_path='yolo11n-pose.pt', camera_source=camera_source)

    # GUI(Frontend) 초기화 (로직 주입)
    app = CollectorGUI(root, logic)

    # 종료 이벤트 처리
    root.protocol("WM_DELETE_WINDOW", app.on_close)

    # 실행
    root.mainloop()


if __name__ == "__main__":
    main()