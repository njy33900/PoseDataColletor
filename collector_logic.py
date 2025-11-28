import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from collections import deque
import datetime
import threading
import time

# ==========================================
# 1. 뼈대 연결 정보 (얼굴 제외)
# ==========================================
# (시작점 인덱스, 끝점 인덱스)
SKELETON_CONNECTIONS = [
    (5, 7), (7, 9),  # 왼팔 (어깨-팔꿈치-손목)
    (6, 8), (8, 10),  # 오른팔
    (11, 13), (13, 15),  # 왼다리 (골반-무릎-발목)
    (12, 14), (14, 16),  # 오른다리
    (5, 6), (11, 12),  # 어깨선, 골반선
    (5, 11), (6, 12)  # 몸통 (좌, 우)
]


# ==========================================
# 2. 쓰레드 기반 카메라 (딜레이 제거)
# ==========================================
class ThreadedCamera:
    def __init__(self, source=0):
        self.capture = cv2.VideoCapture(source)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.status = False
        self.frame = None
        self.lock = threading.Lock()
        self.running = True

        self.status, self.frame = self.capture.read()
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True
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
# 3. 메인 로직 클래스
# ==========================================
class DataCollectorLogic:
    def __init__(self, model_path='yolo11n-pose.pt', camera_source=0, seq_length=30):
        # 최적화 설정
        self.skip_frames = 2
        self.resize_width = 480
        self.resize_height = 360

        self.seq_length = seq_length
        self.is_recording = False
        self.current_label = 0
        self.dataset = []

        self.sequence_buffer = deque(maxlen=seq_length)
        self.smooth_buffer = deque(maxlen=3)
        self.last_valid_pose = np.zeros((17, 2))

        # 캐싱 변수
        self.frame_count = 0
        self.cached_keypoints = None
        self.cached_box = None
        self.cached_conf = None

        print(f"DEBUG: 모델 로딩 중... ({model_path})")
        self.model = YOLO(model_path)

        print("DEBUG: 카메라 쓰레드 시작...")
        self.cap = ThreadedCamera(camera_source)
        print("DEBUG: 초기화 완료")

    def start_recording(self, label_idx):
        self.is_recording = True
        self.current_label = label_idx
        self.sequence_buffer.clear()

    def stop_recording(self):
        self.is_recording = False

    def save_csv(self):
        if not self.dataset:
            return False, "저장할 데이터가 없습니다."
        filename = f"pose_data_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        cols = [f'v{i}' for i in range(self.seq_length * 34)] + ['label']
        df = pd.DataFrame(self.dataset, columns=cols)
        try:
            df.to_csv(filename, index=False)
            return True, f"{filename}\n(총 {len(df)}개)"
        except Exception as e:
            return False, str(e)

    def _preprocess_keypoints(self, keypoints, confidences):
        current_kp = keypoints.copy()
        for i in range(17):
            if confidences[i] < 0.5 or (current_kp[i][0] == 0 and current_kp[i][1] == 0):
                current_kp[i] = self.last_valid_pose[i]
            else:
                self.last_valid_pose[i] = current_kp[i]

        hip_center = (current_kp[11] + current_kp[12]) / 2
        normalized_kp = current_kp - hip_center
        return normalized_kp.flatten(), current_kp

    def process_frame(self):
        ret, frame = self.cap.get_frame()

        if not ret or frame is None:
            return False, None, len(self.dataset)

        # 리사이즈
        frame = cv2.resize(frame, (self.resize_width, self.resize_height))
        draw_frame = frame.copy()

        self.frame_count += 1

        # 추론 (프레임 스킵)
        run_inference = (self.frame_count % self.skip_frames == 0)

        if run_inference:
            results = self.model(frame, verbose=False, conf=0.5)
            if results[0].keypoints is not None and len(results[0].keypoints.xy) > 0:
                self.cached_keypoints = results[0].keypoints.xy.cpu().numpy()[0]
                self.cached_conf = results[0].keypoints.conf.cpu().numpy()[0]
                self.cached_box = results[0].boxes.xyxy.cpu().numpy()[0]
            else:
                self.cached_keypoints = None

        # 결과 처리
        if self.cached_keypoints is not None:
            raw_kp = self.cached_keypoints
            conf = self.cached_conf
            box = self.cached_box

            if run_inference:
                self.smooth_buffer.append(raw_kp)

            if len(self.smooth_buffer) > 0:
                smoothed_kp = np.mean(list(self.smooth_buffer), axis=0)
                processed_data, display_kp = self._preprocess_keypoints(smoothed_kp, conf)

                # --- [시각화 수정 부분] ---

                # 박스 그리기
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(draw_frame, (x1, y1), (x2, y2), (0, 200, 0), 2)

                # 와이어프레임(선) 그리기
                for start_idx, end_idx in SKELETON_CONNECTIONS:
                    pt1 = display_kp[start_idx]
                    pt2 = display_kp[end_idx]

                    # 두 점 모두 화면 안에 있고 좌표가 유효할 때만 그림
                    if pt1[0] > 0 and pt1[1] > 0 and pt2[0] > 0 and pt2[1] > 0:
                        cv2.line(draw_frame,
                                 (int(pt1[0]), int(pt1[1])),
                                 (int(pt2[0]), int(pt2[1])),
                                 (0, 255, 0), 2)  # 초록색 선

                # 관절(점) 그리기 (얼굴 제외)
                for i, (x, y) in enumerate(display_kp):
                    if i <= 4: continue  # 0~4번(코, 눈, 귀)은 건너뜀

                    if x > 0 and y > 0:
                        # 관절마다 색깔 다르게 하려면 여기서 조건문 추가 가능
                        cv2.circle(draw_frame, (int(x), int(y)), 4, (0, 255, 255), -1)  # 노란색 점

                # --- 데이터 수집 로직 ---
                if self.is_recording:
                    self.sequence_buffer.append(processed_data)
                    if len(self.sequence_buffer) == self.seq_length:
                        seq_flat = np.array(self.sequence_buffer).flatten().tolist()
                        seq_flat.append(self.current_label)
                        self.dataset.append(seq_flat)

                        cv2.circle(draw_frame, (20, 20), 8, (0, 0, 255), -1)
                        cv2.putText(draw_frame, "REC", (35, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        return True, draw_frame, len(self.dataset)

    def release(self):
        self.cap.stop()