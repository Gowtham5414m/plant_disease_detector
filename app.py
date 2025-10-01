from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import os
from werkzeug.security import generate_password_hash, check_password_hash
from PIL import Image

app = Flask(__name__)
CORS(app)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
db = SQLAlchemy(app)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)

MODEL_PATH = os.path.join('models', 'plant_disease_cnn.h5')
model = tf.keras.models.load_model(MODEL_PATH)
IMG_SIZE = (128, 128)
CLASS_NAMES = sorted(os.listdir('../dataset/train'))

def create_tables():
    with app.app_context():
        db.create_all()

create_tables()

@app.route('/', methods=['GET'])
def home():
    return "Plant Disease Detector Backend is running."

@app.route('/signup', methods=['POST'])
def signup():
    print("Signup route called")
    try:
        data = request.json
        print("Received signup data:", data)
        email = data.get('email')
        password = data.get('password')
        if not email or not password:
            return jsonify({'message': 'Email and password are required'}), 400

        if User.query.filter_by(email=email).first():
            return jsonify({'message': 'User already exists'}), 400
        
        hashed_password = generate_password_hash(password, method='pbkdf2:sha256')

        user = User(email=email, password=hashed_password)
        db.session.add(user)
        db.session.commit()
        return jsonify({'message': 'Signup successful'}), 200

    except Exception as e:
        print("Signup error:", str(e))
        return jsonify({'message': 'Error during signup', 'error': str(e)}), 500

@app.route('/login', methods=['POST'])
def login():
    print("Login route called")
    try:
        data = request.json
        print("Received login data:", data)
        email = data.get('email')
        password = data.get('password')
        if not email or not password:
            return jsonify({'message': 'Email and password are required'}), 400

        user = User.query.filter_by(email=email).first()
        if not user or not check_password_hash(user.password, password):
            return jsonify({'message': 'Invalid credentials'}), 401
        
        return jsonify({'message': 'Login successful'}), 200

    except Exception as e:
        print("Login error:", str(e))
        return jsonify({'message': 'Error during login', 'error': str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    print("Predict route called")
    if 'file' not in request.files:
        print("No file part in request")
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    try:
        img = Image.open(file.stream)
        print("Image opened successfully")
    except Exception:
        print("Invalid image file sent")
        return jsonify({'error': 'Only leaf images are allowed'}), 400

    img = img.resize(IMG_SIZE)
    img_array = np.array(img) / 255.0
    
    if img_array.ndim == 2 or img_array.shape[-1] != 3:
        print("Image is not RGB or is grayscale")
        return jsonify({'error': 'Only leaf images are allowed'}), 400
    
    img_array = np.expand_dims(img_array, axis=0)
    preds = model.predict(img_array)
    pred_idx = int(np.argmax(preds[0]))
    confidence = float(preds[0][pred_idx])
    pred_class = CLASS_NAMES[pred_idx]
    print(f"Prediction: {pred_class} with confidence {confidence}")

    if confidence < 0.5:
        return jsonify({'disease': 'Unknown disease', 'confidence': confidence}), 200

    return jsonify({'disease': pred_class, 'confidence': confidence}), 200

if __name__ == '__main__':
    app.run(debug=True)
