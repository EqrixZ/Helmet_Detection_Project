import os
import cv2
from flask import Flask, render_template, Response, request, redirect, url_for, send_from_directory
from ultralytics import YOLO

app = Flask(__name__)

model = YOLO("best.pt")
names = model.model.names

os.makedirs('uploads', exist_ok=True)
PREDICT_FOLDER = os.path.join('runs', 'predict')
os.makedirs(PREDICT_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start_webcam')
def start_webcam():
    return render_template('webcam.html')

def detect_objects_from_webcam():
    count = 0
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret: break
        count += 1  
        if count % 2 != 0: continue
           
        frame = cv2.resize(frame, (1020, 600))
        results = model.track(frame, persist=True)

        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.int().cpu().tolist()
            class_ids = results[0].boxes.cls.int().cpu().tolist()
            track_ids = results[0].boxes.id.int().cpu().tolist()

            for box, class_id, track_id in zip(boxes, class_ids, track_ids):
                    c = names[class_id]
                    x1, y1, x2, y2 = box
                    
                    if c == 'Helmet':
                        color = (0, 255, 0)
                    elif c == 'Head':
                        color = (0, 0, 255)
                    else:
                        color = (255, 255, 0)
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f'{track_id} - {c}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/webcam_feed')
def webcam_feed():
    return Response(detect_objects_from_webcam(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/upload', methods=['POST'])
def upload_video():
    if 'file' not in request.files: return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '': return redirect(request.url)

    file_path = os.path.join('uploads', file.filename)
    file.save(file_path)

    return redirect(url_for('play_video', filename=file.filename))

@app.route('/uploads/<filename>')
def play_video(filename):
    return render_template('play_video.html', filename=filename)

@app.route('/video/<path:filename>')
def send_video(filename):
    return send_from_directory('uploads', filename)

def detect_objects_from_video(video_path, filename):
    cap = cv2.VideoCapture(video_path)
    
    out_width, out_height = 1020, 600
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    save_path = os.path.join(PREDICT_FOLDER, f"predict_{filename}")
    out = cv2.VideoWriter(save_path, fourcc, fps / 2, (out_width, out_height))
    
    count = 0
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            count += 1
            if count % 2 != 0: continue
            
            frame = cv2.resize(frame, (out_width, out_height))
            results = model.track(frame, persist=True)

            if results[0].boxes is not None and results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.int().cpu().tolist()
                class_ids = results[0].boxes.cls.int().cpu().tolist()
                track_ids = results[0].boxes.id.int().cpu().tolist()

                for box, class_id, track_id in zip(boxes, class_ids, track_ids):
                    c = names[class_id]
                    x1, y1, x2, y2 = box
                    
                    if c == 'Helmet':
                        color = (0, 255, 0)
                    elif c == 'Head':
                        color = (0, 0, 255)
                    else:
                        color = (255, 255, 0)
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f'{track_id} - {c}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            out.write(frame)

            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    finally:
        cap.release()
        out.release()

@app.route('/video_feed/<filename>')
def video_feed(filename):
    video_path = os.path.join('uploads', filename)
    return Response(detect_objects_from_video(video_path, filename), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run('0.0.0.0', debug=True, port=8080)