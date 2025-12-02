import cv2
import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from collector_logic import DataCollectorLogic
import threading
import time

# 전역 로직 인스턴스
logic = None

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class LabelRequest(BaseModel):
    label: int


def init_logic(camera_source):
    global logic
    if logic is None:
        logic = DataCollectorLogic(model_path='yolo11n-pose.pt', camera_source=camera_source)


# ---------------------------------------------------------
# API 엔드포인트
# ---------------------------------------------------------

# 녹화시작
@app.post("/control/start")
def start_recording(req: LabelRequest):
    if logic:
        logic.start_recording(req.label)
        return {"status": "started", "label": req.label}
    return {"status": "error", "message": "Logic not initialized"}

# 저장
@app.post("/control/save")
def save_data():
    if logic:
        success, msg = logic.save_csv()
        return {"success": success, "message": msg}
    return {"success": False, "message": "Logic not initialized"}

# 상태표시
@app.get("/status")
def get_status():
    if logic:
        return {
            "count": len(logic.dataset),
            "is_recording": logic.is_recording,
            "label": logic.current_label
        }
    return {"count": 0, "is_recording": False}

# 되돌리기(이전 기록 삭제)
@app.post("/control/undo")
def undo_recording():
    """마지막 녹화 취소 요청"""
    if logic:
        success, msg = logic.undo_last_recording()
        return {"success": success, "message": msg}
    return {"success": False, "message": "Logic not initialized"}

# 프레임 생성
def generate_frames():
    while True:
        if logic:
            # 로직 처리 (최신 프레임 가져오기)
            ret, frame, _, _ = logic.process_frame()

            if ret:
                # [최적화 1] JPEG 압축률 조정 (기본 95 -> 50)
                # 화질은 조금 떨어지지만 전송 속도가 3~5배 빨라짐
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 50]

                # [최적화 2] 인코딩
                ret, buffer = cv2.imencode('.jpg', frame, encode_param)

                if ret:
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

            # [최적화 3] 과도한 CPU 사용 방지 및 네트워크 숨통 트기 (약 30FPS 제한)
            time.sleep(0.03)
        else:
            time.sleep(0.1)


@app.get("/video_feed")
def video_feed():
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")


def run_api(camera_source=0):
    init_logic(camera_source)
    print("🚀 웹 모드 시작: http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)