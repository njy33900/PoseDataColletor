import tkinter as tk
from tkinter import ttk, messagebox
import cv2
from PIL import Image, ImageTk


class CollectorGUI:
    def __init__(self, root, logic_controller):
        self.root = root
        self.logic = logic_controller

        self.root.title("AI Pose Data Collector (Modular)")
        self.root.geometry("1100x600")

        self._init_ui()

        # 영상 업데이트 루프 시작
        self.update_video_loop()

    def _init_ui(self):
        # 좌측: 비디오 패널
        self.video_frame = tk.Frame(self.root, bg="black", width=800, height=600)
        self.video_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.lbl_video = tk.Label(self.video_frame, bg="black")
        self.lbl_video.pack(fill=tk.BOTH, expand=True)

        # 우측: 컨트롤 패널
        self.control_frame = tk.Frame(self.root, bg="#f0f0f0", width=300)
        self.control_frame.pack(side=tk.RIGHT, fill=tk.Y)

        # --- 컨트롤 구성 ---
        tk.Label(self.control_frame, text="데이터 수집 제어", font=("Arial", 16, "bold"), bg="#f0f0f0").pack(pady=20)

        # 라벨 선택
        tk.Label(self.control_frame, text="1. 데이터 라벨 선택", font=("Arial", 12), bg="#f0f0f0").pack(anchor="w", padx=20)
        self.class_var = tk.IntVar(value=0)
        modes = [("Neutral (정지)", 0), ("Movement (이동)", 1), ("Suspicious (위험)", 2)]
        for text, val in modes:
            ttk.Radiobutton(self.control_frame, text=text, variable=self.class_var, value=val).pack(anchor="w", padx=30,
                                                                                                    pady=5)

        ttk.Separator(self.control_frame, orient='horizontal').pack(fill='x', pady=20, padx=10)

        # 버튼
        self.btn_start = tk.Button(self.control_frame, text="🔴 캡쳐 시작", bg="#ffcccc", font=("Arial", 12),
                                   command=self.on_start)
        self.btn_start.pack(fill='x', padx=20, pady=5)

        self.btn_stop = tk.Button(self.control_frame, text="⬛ 캡쳐 종료", bg="#cccccc", font=("Arial", 12),
                                  state="disabled", command=self.on_stop)
        self.btn_stop.pack(fill='x', padx=20, pady=5)

        # 상태 표시
        self.lbl_status = tk.Label(self.control_frame, text="대기 중...", fg="gray", bg="#f0f0f0", font=("Arial", 10))
        self.lbl_status.pack(pady=10)
        self.lbl_count = tk.Label(self.control_frame, text="수집된 데이터: 0개", font=("Arial", 12, "bold"), bg="#f0f0f0")
        self.lbl_count.pack(pady=5)

        ttk.Separator(self.control_frame, orient='horizontal').pack(fill='x', pady=20, padx=10)

        # 저장
        self.btn_save = tk.Button(self.control_frame, text="💾 CSV 저장", bg="#ccffcc", font=("Arial", 12, "bold"),
                                  command=self.on_save)
        self.btn_save.pack(fill='x', padx=20, pady=20)

    def update_video_loop(self):
        # Logic에게 프레임 처리 요청
        ret, frame, data_count = self.logic.process_frame()

        if ret:
            # OpenCV(BGR) -> Tkinter(RGB) 변환
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb_image)
            # 화면 크기에 맞춰 리사이즈 (선택사항)
            # img = img.resize((800, 600))
            imgtk = ImageTk.PhotoImage(image=img)

            self.lbl_video.imgtk = imgtk
            self.lbl_video.configure(image=imgtk)

            # 카운터 업데이트 (부하 줄이기 위해 가끔 업데이트해도 됨)
            self.lbl_count.config(text=f"수집된 데이터: {data_count}개")

        # 10ms 후 반복
        self.root.after(10, self.update_video_loop)

    def on_start(self):
        label_idx = self.class_var.get()
        label_name = ["Neutral", "Movement", "Suspicious"][label_idx]

        self.logic.start_recording(label_idx)

        self.btn_start.config(state="disabled", bg="#cccccc")
        self.btn_stop.config(state="normal", bg="#ffcccc")
        self.lbl_status.config(text=f"녹화 중... [{label_name}]", fg="red")

    def on_stop(self):
        self.logic.stop_recording()

        self.btn_start.config(state="normal", bg="#ffcccc")
        self.btn_stop.config(state="disabled", bg="#cccccc")
        self.lbl_status.config(text="녹화 중지됨", fg="blue")

    def on_save(self):
        success, msg = self.logic.save_csv()
        if success:
            messagebox.showinfo("저장 완료", msg)
        else:
            messagebox.showwarning("실패", msg)

    def on_close(self):
        self.logic.release()
        self.root.destroy()