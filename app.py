from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
from keras.models import load_model
from ultralytics import YOLO

# Konfigurasi
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load model
model_cnn = load_model('trained_models/model_deteksi_daun_mawar_tf.h5')
model_yolo = YOLO('best.pt')

# Fungsi cek file
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Fungsi klasifikasi daun mawar
def is_mawar(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (128, 128))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)
    pred = model_cnn.predict(img)[0][0]
    return pred > 0.5

# Halaman utama
@app.route('/')
def index():
    return render_template('index.html')

# Prediksi deteksi penyakit
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file'}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Cek daun mawar
    if not is_mawar(filepath):
        return jsonify({'error': 'Gambar bukan daun mawar'}), 400

    # Prediksi YOLO
    results = model_yolo.predict(filepath)[0]
    boxes = []

    if results.boxes is not None:
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            label = model_yolo.names[cls_id]

            boxes.append({
                "x": x1,
                "y": y1,
                "width": x2 - x1,
                "height": y2 - y1,
                "label": label,
                "confidence": conf
            })

    return jsonify({
        'image_url': f'/static/uploads/{filename}',
        'boxes': boxes
    })

if __name__ == '__main__':
    app.run(debug=True)
