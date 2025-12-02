import tkinter as tk
from tkinter import ttk, messagebox
import cv2
from PIL import Image, ImageTk


class CollectorGUI:
    def __init__(self, root, logic_controller):
        self.root = root
        self.logic = logic_controller

        self.root.title("AI Pose Data Collector (Desktop GUI)")
        self.root.geometry("1100x700")  # 버튼 추가로 높이 약간 증가

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
        tk.Label(self.control_frame, text="1. 행동 라벨 선택", font=("Arial", 12), bg="#f0f0f0").pack(anchor="w", padx=20)
        self.class_var = tk.IntVar(value=0)
        modes = [("Neutral (정지)", 0), ("Movement (이동)", 1), ("Suspicious (위험)", 2)]
        for text, val in modes:
            ttk.Radiobutton(self.control_frame, text=text, variable=self.class_var, value=val).pack(anchor="w", padx=30,
                                                                                                    pady=5)

        ttk.Separator(self.control_frame, orient='horizontal').pack(fill='x', pady=20, padx=10)

        # 녹화 제어
        tk.Label(self.control_frame, text="2. 녹화 제어", font=("Arial", 12), bg="#f0f0f0").pack(anchor="w", padx=20)

        # 시작 버튼
        self.btn_start = tk.Button(self.control_frame, text="🔴 3초간 녹화 시작", bg="#ffcccc", font=("Arial", 12, "bold"),
                                   height=2, command=self.on_start)
        self.btn_start.pack(fill='x', padx=20, pady=10)

        # [추가됨] 되돌리기 버튼
        self.btn_undo = tk.Button(self.control_frame, text="↩️ 방금 녹화 취소 (Undo)", bg="#ffeb99",
                                  font=("Arial", 11, "bold"), command=self.on_undo)
        self.btn_undo.pack(fill='x', padx=20, pady=5)

        # 상태 표시
        self.lbl_status = tk.Label(self.control_frame, text="대기 중...", fg="gray", bg="#f0f0f0", font=("Arial", 10))
        self.lbl_status.pack(pady=10)

        self.lbl_count = tk.Label(self.control_frame, text="수집된 데이터: 0개", font=("Arial", 14, "bold"), bg="#f0f0f0")
        self.lbl_count.pack(pady=10)

        ttk.Separator(self.control_frame, orient='horizontal').pack(fill='x', pady=20, padx=10)

        # 저장
        self.btn_save = tk.Button(self.control_frame, text="💾 CSV 파일로 저장", bg="#ccffcc", font=("Arial", 12, "bold"),
                                  command=self.on_save)
        self.btn_save.pack(fill='x', padx=20, pady=20)

    def update_video_loop(self):
        # Logic에게 프레임 처리 요청
        ret, frame, data_count, is_rec = self.logic.process_frame()

        if ret:
            # OpenCV(BGR) -> Tkinter(RGB)
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb_image)
            imgtk = ImageTk.PhotoImage(image=img)

            self.lbl_video.imgtk = imgtk  # 참조 유지
            self.lbl_video.configure(image=imgtk)

            # 카운터 업데이트
            self.lbl_count.config(text=f"수집된 데이터: {data_count}개")

            # [상태 관리] 녹화가 끝났으면 버튼 다시 활성화
            if not is_rec and self.btn_start['state'] == 'disabled':
                self.btn_start.config(state="normal", bg="#ffcccc", text="🔴 3초간 녹화 시작")
                self.lbl_status.config(text="녹화 완료 (저장됨)", fg="blue")
                self.btn_undo.config(state="normal")  # 녹화 끝나면 Undo 가능

        # 30ms 후 반복
        self.root.after(30, self.update_video_loop)

    def on_start(self):
        label_idx = self.class_var.get()
        label_name = ["Neutral", "Movement", "Suspicious"][label_idx]

        self.logic.start_recording(label_idx)

        # 버튼 비활성화 (중복 클릭 방지)
        self.btn_start.config(state="disabled", bg="#cccccc", text="녹화 중...")
        self.btn_undo.config(state="disabled")  # 녹화 중엔 Undo 불가
        self.lbl_status.config(text=f"녹화 중... [{label_name}]", fg="red")

    def on_undo(self):
        """되돌리기 버튼 핸들러"""
        if not self.logic.dataset:
            messagebox.showwarning("경고", "삭제할 데이터가 없습니다.")
            return

        if messagebox.askyesno("확인", "방금 기록한 데이터와 영상을 정말 삭제하시겠습니까?"):
            success, msg = self.logic.undo_last_recording()
            if success:
                messagebox.showinfo("성공", msg)
                # 카운트 즉시 갱신
                self.lbl_count.config(text=f"수집된 데이터: {len(self.logic.dataset)}개")
            else:
                messagebox.showerror("오류", msg)

    def on_save(self):
        success, msg = self.logic.save_csv()
        if success:
            messagebox.showinfo("저장 완료", msg)
        else:
            messagebox.showwarning("실패", msg)

    def on_close(self):
        self.logic.release()
        self.root.destroy()