import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from collections import deque
import datetime
import threading
import time
import torch
import os

# ==========================================
# 뼈대 연결 정보
# ==========================================
SKELETON_CONNECTIONS = [
    (0, 5), (0, 6),  # 목 (코-어깨)
    (5, 7), (7, 9),  # 왼팔
    (6, 8), (8, 10),  # 오른팔
    (11, 13), (13, 15),  # 왼다리
    (12, 14), (14, 16),  # 오른다리
    (5, 6), (11, 12),  # 몸통 가로
    (5, 11), (6, 12)  # 몸통 세로
]


# ==========================================
# 쓰레드 기반 카메라 (딜레이 제거)
# ==========================================
class ThreadedCamera:
    def __init__(self, source=0):
        self.capture = cv2.VideoCapture(source)
        # 하드웨어 버퍼 최소화
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.lock = threading.Lock()
        self.running = True
        self.status = False
        self.frame = None

        # 초기 프레임 읽기
        self.status, self.frame = self.capture.read()

        # 백그라운드 업데이트 시작
        self.thread = threading.Thread(target=self.update, args=(), daemon=True)
        self.thread.start()

    def update(self):
        while self.running:
            if self.capture.isOpened():
                status, frame = self.capture.read()
                with self.lock:
                    self.status = status
                    self.frame = frame
            else:
                time.sleep(0.01)

    def get_frame(self):
        with self.lock:
            return self.status, self.frame

    def stop(self):
        self.running = False
        self.thread.join()
        self.capture.release()


# ==========================================
# 데이터 수집 로직 (Undo 기능 포함)
# ==========================================
class DataCollectorLogic:
    def __init__(self, model_path='yolo11n-pose.pt', camera_source=0, seq_length=30):
        # --- 설정 ---
        self.target_duration = 3.0
        self.collect_interval = 3
        self.resize_dims = (320, 240)

        self.persistence_threshold = 10
        self.miss_count = 0

        # GPU 설정
        self.device = '0' if torch.cuda.is_available() else 'cpu'
        print(f"DEBUG: 추론 장치 -> {self.device}")

        self.seq_length = seq_length
        self.dataset = []

        # [추가] 녹화된 비디오 경로 추적용 리스트 (Undo 기능용)
        self.video_paths = []

        self.is_recording = False
        self.start_time = 0
        self.current_label = 0
        self.current_video_path = None

        self.video_writer = None
        self.csv_save_dir = "data"
        self.video_base_dir = "video"

        # 버퍼
        self.temp_sequence = []
        self.smooth_buffer = deque(maxlen=3)
        self.last_valid_pose = np.zeros((17, 2))

        # 모델 로드
        print(f"DEBUG: 모델 로딩 중... ({model_path})")
        self.model = YOLO(model_path)
        if self.device == '0': self.model.to('cuda')

        # 카메라 시작
        print("DEBUG: 카메라 쓰레드 시작...")
        self.cap = ThreadedCamera(camera_source)
        self.frame_count = 0

        self.cached_keypoints = None
        self.cached_box = None
        self.cached_conf = None

        # 폴더 생성
        if not os.path.exists(self.csv_save_dir):
            os.makedirs(self.csv_save_dir)

    def start_recording(self, label_idx):
        """녹화 시작: 비디오 파일 생성"""
        self.is_recording = True
        self.current_label = label_idx
        self.start_time = time.time()
        self.temp_sequence = []

        # 날짜/시간 정보
        now = datetime.datetime.now()
        date_str = now.strftime('%Y%m%d')
        time_str = now.strftime('%H%M%S')

        label_map = {0: "Neutral", 1: "Movement", 2: "Suspicious"}
        label_str = label_map.get(label_idx, "Unknown")

        # 폴더 생성 (video/YYYYMMDD/)
        daily_video_dir = os.path.join(self.video_base_dir, date_str)
        if not os.path.exists(daily_video_dir):
            os.makedirs(daily_video_dir)

        # 파일명 생성
        video_filename = f"video_{label_str}_{time_str}.mp4"
        self.current_video_path = os.path.join(daily_video_dir, video_filename)

        # VideoWriter 초기화
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(self.current_video_path, fourcc, 30.0, self.resize_dims)

        print(f"DEBUG: 녹화 시작 -> {self.current_video_path}")

    def stop_recording(self):
        """강제 종료 시"""
        self.is_recording = False
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None

    def undo_last_recording(self):
        """[핵심] 마지막 녹화 데이터 및 영상 삭제"""
        if not self.dataset:
            return False, "삭제할 데이터가 없습니다."

        # 데이터셋에서 마지막 항목 제거
        self.dataset.pop()

        # 비디오 파일 삭제
        deleted_file_name = "Unknown"
        if self.video_paths:
            last_path = self.video_paths.pop()
            if os.path.exists(last_path):
                try:
                    os.remove(last_path)
                    deleted_file_name = os.path.basename(last_path)
                    print(f"DEBUG: 파일 삭제됨 -> {last_path}")
                except Exception as e:
                    print(f"ERROR: 파일 삭제 실패 - {e}")

        return True, f"되돌리기 성공!\n삭제된 영상: {deleted_file_name}\n남은 데이터: {len(self.dataset)}개"

    def save_csv(self):
        if not self.dataset:
            return False, "저장할 데이터가 없습니다."

        filename = f"pose_data_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        full_path = os.path.join(self.csv_save_dir, filename)

        cols = [f'v{i}' for i in range(self.seq_length * 34)] + ['label']
        df = pd.DataFrame(self.dataset, columns=cols)

        try:
            df.to_csv(full_path, index=False)
            return True, f"CSV 저장 완료!\n{full_path}\n(총 {len(df)}개 시퀀스)"
        except Exception as e:
            return False, str(e)

    def _preprocess_keypoints(self, keypoints, confidences):
        current_kp = keypoints.copy()

        # 결측치 보정
        for i in range(17):
            if confidences[i] < 0.5 or (current_kp[i][0] == 0 and current_kp[i][1] == 0):
                current_kp[i] = self.last_valid_pose[i]
            else:
                self.last_valid_pose[i] = current_kp[i]

        # 동적 앵커 포인트 (하체/측면 가림 대응)
        anchor_x, anchor_y = 0, 0
        valid_anchor_found = False

        # 골반
        left_hip_ok = confidences[11] > 0.5
        right_hip_ok = confidences[12] > 0.5
        if left_hip_ok and right_hip_ok:
            anchor_x = (current_kp[11][0] + current_kp[12][0]) / 2
            anchor_y = (current_kp[11][1] + current_kp[12][1]) / 2
            valid_anchor_found = True
        elif left_hip_ok:
            anchor_x, anchor_y = current_kp[11]
            valid_anchor_found = True
        elif right_hip_ok:
            anchor_x, anchor_y = current_kp[12]
            valid_anchor_found = True

        # 어깨
        if not valid_anchor_found:
            left_sh_ok = confidences[5] > 0.5
            right_sh_ok = confidences[6] > 0.5
            if left_sh_ok and right_sh_ok:
                anchor_x = (current_kp[5][0] + current_kp[6][0]) / 2
                anchor_y = (current_kp[5][1] + current_kp[6][1]) / 2
                valid_anchor_found = True
            elif left_sh_ok:
                anchor_x, anchor_y = current_kp[5]
                valid_anchor_found = True
            elif right_sh_ok:
                anchor_x, anchor_y = current_kp[6]
                valid_anchor_found = True

        # 코
        if not valid_anchor_found:
            if confidences[0] > 0.5:
                anchor_x, anchor_y = current_kp[0]
            else:
                anchor_x, anchor_y = 0, 0

                # 상대 좌표 변환
        if anchor_x != 0:
            normalized_kp = current_kp - [anchor_x, anchor_y]
        else:
            normalized_kp = current_kp

        return normalized_kp.flatten(), current_kp

    def process_frame(self):
        ret, frame = self.cap.get_frame()
        if not ret or frame is None:
            return False, None, len(self.dataset), self.is_recording

        frame = cv2.resize(frame, self.resize_dims)
        draw_frame = frame.copy()
        self.frame_count += 1

        # --- 3초 타이머 및 종료 로직 ---
        timer_text = ""
        if self.is_recording:
            elapsed = time.time() - self.start_time
            remaining = self.target_duration - elapsed

            if remaining <= 0:
                # [종료 시점]
                remaining = 0
                self.is_recording = False

                # 비디오 저장 종료
                if self.video_writer:
                    self.video_writer.release()
                    self.video_writer = None

                # 데이터 유효성 검사 및 저장
                if len(self.temp_sequence) >= self.seq_length:
                    final_seq = self.temp_sequence[:self.seq_length]
                    seq_flat = np.array(final_seq).flatten().tolist()
                    seq_flat.append(self.current_label)

                    self.dataset.append(seq_flat)

                    # 성공적으로 저장된 비디오 경로 기록 (Undo를 위함)
                    self.video_paths.append(self.current_video_path)

                    print(f"DEBUG: 저장 완료. (Total: {len(self.dataset)})")
                else:
                    # 데이터 부족 시 비디오 삭제 (쓰레기 파일 방지)
                    if os.path.exists(self.current_video_path):
                        try:
                            os.remove(self.current_video_path)
                            print("DEBUG: 데이터 부족으로 영상 자동 삭제됨")
                        except:
                            pass

            timer_text = f"{remaining:.2f}s"

        # --- YOLO 추론 ---
        run_inference = (self.frame_count % self.collect_interval == 0)

        if run_inference:
            results = self.model(frame, verbose=False, conf=0.5, device=self.device)
            if results[0].keypoints is not None and len(results[0].keypoints.xy) > 0:
                self.cached_keypoints = results[0].keypoints.xy.cpu().numpy()[0]
                self.cached_conf = results[0].keypoints.conf.cpu().numpy()[0]
                self.cached_box = results[0].boxes.xyxy.cpu().numpy()[0]
                self.miss_count = 0
            else:
                self.miss_count += 1
                if self.miss_count > self.persistence_threshold:
                    self.cached_keypoints = None
                    self.cached_box = None

        # --- 그리기 ---
        if self.cached_keypoints is not None:
            raw_kp = self.cached_keypoints
            conf = self.cached_conf if self.cached_conf is not None else np.ones(17)

            if run_inference and self.miss_count == 0:
                self.smooth_buffer.append(raw_kp)

            if len(self.smooth_buffer) > 0:
                smoothed_kp = np.mean(list(self.smooth_buffer), axis=0)
                processed_data, display_kp = self._preprocess_keypoints(smoothed_kp, conf)

                # 박스
                if self.cached_box is not None:
                    x1, y1, x2, y2 = map(int, self.cached_box)
                    cv2.rectangle(draw_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # 와이어프레임
                for start, end in SKELETON_CONNECTIONS:
                    p1, p2 = display_kp[start], display_kp[end]
                    if p1[0] > 0 and p2[0] > 0:
                        cv2.line(draw_frame, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (0, 255, 0), 2)

                # 관절 점 (코 포함, 눈/귀 제외)
                for i, (x, y) in enumerate(display_kp):
                    if i != 0 and i <= 4: continue
                    if x > 0:
                        color = (0, 255, 255)
                        if i == 0: color = (255, 0, 255)
                        cv2.circle(draw_frame, (int(x), int(y)), 4, color, -1)

                if self.is_recording:
                    self.temp_sequence.append(processed_data)

        # --- 화면 오버레이 ---
        if self.is_recording:
            cv2.rectangle(draw_frame, (0, 0), (self.resize_dims[0], self.resize_dims[1]), (0, 0, 255), 4)
            cv2.putText(draw_frame, f"REC {timer_text}", (100, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            # 현재 프레임을 비디오 파일에 기록
            if self.video_writer is not None:
                self.video_writer.write(draw_frame)

        return True, draw_frame, len(self.dataset), self.is_recording

    def release(self):
        if self.video_writer: self.video_writer.release()
        self.cap.stop()