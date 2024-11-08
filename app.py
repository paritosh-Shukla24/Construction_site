from ultralytics import YOLO
import cv2
import numpy as np
import base64
from flask import Flask, render_template
from flask_socketio import SocketIO
import threading
import queue
import time
from datetime import datetime
import json

# Initialize Flask and SocketIO
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Global variables
frame_queue = queue.Queue(maxsize=10)
result_queue = queue.Queue(maxsize=10)
processing_time = queue.Queue(maxsize=1)
processing_time.put(0)

# Load YOLO model
model = YOLO('best.pt')

# Label mappings with colors (BGR format)
class_config = {
    0: {"name": "Helmet", "color": (0, 255, 0)},      # Green
    1: {"name": "Mask", "color": (255, 191, 0)},       # Deep Sky Blue
    2: {"name": "NO-Helmet", "color": (0, 0, 255)},   # Red
    3: {"name": "NO-Mask", "color": (255, 0, 0)},      # Blue
    4: {"name": "NO-Safety Vest", "color": (128, 0, 128)}, # Purple
    5: {"name": "Person", "color": (255, 165, 0)},     # Orange
    6: {"name": "Safety Cone", "color": (0, 165, 255)}, # Orange-Yellow
    7: {"name": "Safety Vest", "color": (0, 255, 255)}, # Yellow
    8: {"name": "machinery", "color": (139, 69, 19)},   # Brown
    9: {"name": "vehicle", "color": (128, 128, 128)}    # Gray
}

# Separate violations for special handling
violation_classes = {'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest'}

class VideoCamera:
    def __init__(self, source=0):
        self.video = cv2.VideoCapture(source)
        self.video.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.video.set(cv2.CAP_PROP_FPS, 30)
        self.video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.stopped = False
        
    def start(self):
        threading.Thread(target=self.update, daemon=True).start()
        return self
    
    def update(self):
        while not self.stopped:
            if not frame_queue.full():
                ret, frame = self.video.read()
                if ret:
                    frame_queue.put(frame)
                else:
                    self.stopped = True
            else:
                time.sleep(0.001)
    
    def stop(self):
        self.stopped = True
        self.video.release()

class DetectionProcessor:
    def __init__(self):
        self.stopped = False
        
    def start(self):
        threading.Thread(target=self.process, daemon=True).start()
        return self
        
    def process(self):
        while not self.stopped:
            if not frame_queue.empty() and not result_queue.full():
                start_time = time.time()
                
                # Get frame from queue
                frame = frame_queue.get()
                
                # Run detection
                results = model(frame, stream=True)
                
                # Process results
                detections = {
                    'violations': [],
                    'standard': []
                }
                annotated_frame = frame.copy()
                
                for r in results:
                    boxes = r.boxes
                    for box in boxes:
                        # Get box coordinates
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        
                        # Get class and confidence
                        cls = int(box.cls[0])
                        conf = float(box.conf[0])
                        
                        # Get class info
                        class_info = class_config[cls]
                        label = class_info["name"]
                        color = class_info["color"]
                        
                        # Create detection info
                        detection_info = {
                            "label": label,
                            "confidence": round(conf, 2)
                        }
                        
                        # Add to appropriate list
                        if label in violation_classes:
                            detections['violations'].append(detection_info)
                        else:
                            detections['standard'].append(detection_info)
                            
                        # Draw box with class-specific color
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                        
                        # Create label background
                        text = f"{label} {conf:.2f}"
                        (text_width, text_height), _ = cv2.getTextSize(
                            text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                        cv2.rectangle(annotated_frame,
                                    (x1, y1 - text_height - 10),
                                    (x1 + text_width + 10, y1),
                                    color, -1)
                        
                        # Add white text
                        cv2.putText(annotated_frame, text,
                                  (x1 + 5, y1 - 5),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                  (255, 255, 255), 2)
                
                # Encode frame
                _, buffer = cv2.imencode('.jpg', annotated_frame, 
                                       [cv2.IMWRITE_JPEG_QUALITY, 80])
                frame_encoded = base64.b64encode(buffer).decode('utf-8')
                
                # Calculate processing time
                process_time = time.time() - start_time
                processing_time.get()
                processing_time.put(process_time)
                
                # Put results in queue
                result_queue.put({
                    'frame': frame_encoded,
                    'detections': detections
                })
            else:
                time.sleep(0.001)
    
    def stop(self):
        self.stopped = True

@app.route('/')
def index():
    return render_template('index.html', class_config=class_config)

def send_frames():
    while True:
        if not result_queue.empty():
            result = result_queue.get()
            socketio.emit('frame', result)
        socketio.sleep(0.001)

@socketio.on('connect')
def on_connect():
    print('Client connected')
    socketio.start_background_task(send_frames)

if __name__ == '__main__':
    # Start video capture and processing threads
    video_camera = VideoCamera().start()
    processor = DetectionProcessor().start()
    
    # Run the server
    socketio.run(app, host='0.0.0.0', port=5000, debug=False)