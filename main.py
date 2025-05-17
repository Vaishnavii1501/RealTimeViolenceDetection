from flask import Flask, request, render_template, redirect, url_for, Response, jsonify # type: ignore
from werkzeug.utils import secure_filename # type: ignore
import os
import cv2
import numpy as np
from keras.models import load_model # type: ignore
from collections import deque
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import ssl
import base64
import logging
import datetime
import requests

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'

should_process_video = False


@app.route('/start_processing', methods=['POST'])
def start_processing():
    data = request.get_json()
    filename = data['filename']
    global should_process_video
    should_process_video = True
    return jsonify({'message': 'Processing started'})


@app.route('/start_camera', methods=['POST'])
def start_camera():
    global should_process_video
    should_process_video = True
    return jsonify({'message': 'Camera started'})


@app.route('/stop_camera', methods=['POST'])
def stop_camera():
    global should_process_video
    should_process_video = False
    return jsonify({'message': 'Camera stopped'})


@app.route('/')
def upload_file():
    return render_template('index.html')


@app.route('/uploader', methods=['GET', 'POST'])
def upload_video():
    if request.method == 'POST':
        f = request.files['file']
        if f.filename == '':
            return 'No file selected', 400
            
        filename = secure_filename(f.filename)

       
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        f.save(filepath)

        
        return filename


@app.route('/preview/<filename>')
def preview(filename):
    return render_template('preview.html', filename=filename)


@app.route('/video_feed/<filename>')
def video_feed(filename):
    global should_process_video
    if not should_process_video:
        return '', 204
    return Response(generate_frames('static/uploads/' + filename),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/processed_video_feed/<filename>')
def processed_video_feed(filename):
    return Response(generate_frames('static/uploads/' + filename),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/camera_feed')
def camera_feed():
    global should_process_video
    if not should_process_video:
        return '', 204  
    return Response(generate_frames(0),  
                    mimetype='multipart/x-mixed-replace; boundary=frame')


def send_email(subject, body, attachment=None):
    try:
       
        smtp_server = "smtp.gmail.com"
        port = 587  
        username = "sambarevaishnavi20@gmail.com"
        password = "yzos ecdb ntwc hrpl"

        context = ssl.create_default_context()

       
        server = smtplib.SMTP(smtp_server, port)
        server.ehlo()  
        server.starttls(context=context)  
        server.ehlo()  
        server.login(username, password)

        msg = MIMEMultipart()
        msg['From'] = username
        msg['To'] = 'achal.22220332@viit.ac.in'
        msg['Subject'] = subject
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        
        response = requests.get('https://ipinfo.io')
        if response.status_code == 200:
            data = response.json()
            location = f"{data.get('city', 'Unknown city')}, {data.get('region', 'Unknown region')}"
        else:
            location = 'Unknown'

        
        body = (f'Dear Author,\n\nViolence Deteced at {current_time}.\nThe location is: {location}. \n\n'
                f'Please take an action.The current situation given bellow: ')

        msg.attach(MIMEText(body, 'plain'))

        if attachment is not None:
            _, img_encoded = cv2.imencode('.jpg', attachment, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            img_as_bytes = img_encoded.tobytes()

            img_part = MIMEBase('image', "jpeg")
            img_part.set_payload(img_as_bytes)
            encoders.encode_base64(img_part)

            img_part.add_header('Content-Disposition', 'attachment', filename='detected_frame.jpg')
            msg.attach(img_part)

        server.send_message(msg)
        server.quit()

        logging.info('Email sent.')
    except Exception as e:
        logging.error(f'Error sending email: {str(e)}')


def generate_frames(video_path):
    model = load_model("C:\\Users\\ajink\\OneDrive\\Desktop\\CP\\CP\\models\\violence_detection_model.h5")
    image_height, image_width = 96, 96  
    sequence_length = 16
    class_list = ["Violence", "Non-Violence"]

    video_reader = cv2.VideoCapture(video_path)
    fps = video_reader.get(cv2.CAP_PROP_FPS)
    print(f'The video has {fps} frames per second.')

    frames_queue = deque(maxlen=sequence_length)
    predicted_class_name = ''
    predicted_confidence = 0
    violence_count = 0
    mail_sent = False
    violent_frame = None

    while video_reader.isOpened():
       
        global should_process_video
        if not should_process_video:
            
            video_reader.release()
            break
            
        ok, frame = video_reader.read()
        if not ok:
            break

       
        resized_frame = cv2.resize(frame, (image_height, image_width))
        normalized_frame = resized_frame / 255.0  

        frames_queue.append(normalized_frame)

        if len(frames_queue) == sequence_length:
            predicted_labels_probabilities = model.predict(np.expand_dims(frames_queue, axis=0))[0]
            predicted_label = np.argmax(predicted_labels_probabilities)
            predicted_class_name = class_list[predicted_label]
            predicted_confidence = predicted_labels_probabilities[predicted_label]

      
        text = f'{predicted_class_name}: {predicted_confidence:.2f}'
        
        
        text_color = (0, 0, 255) if predicted_class_name == "Violence" else (0, 255, 0)
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)

       
        if predicted_class_name == "Violence":
            violence_count += 1
            
            if violent_frame is None:
                violent_frame = frame.copy()
                
            
            if violence_count >= 3 and not mail_sent:
                send_email('Violence Detected!!!', 'A violent event was detected.', violent_frame)
                mail_sent = True
                print("Violence alert sent!")
        else:
            
            violence_count = 0
            violent_frame = None

        ret, jpeg = cv2.imencode('.jpg', frame)
        frame = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

    video_reader.release()
    cv2.destroyAllWindows()



if __name__ == '__main__':
    app.run(debug=True)