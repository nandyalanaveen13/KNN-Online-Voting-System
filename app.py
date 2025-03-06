
from flask import Flask, render_template, request, redirect, url_for, session, jsonify, Response
import cv2
import numpy as np
import pickle
import os
import sqlite3
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash
from sklearn.neighbors import KNeighborsClassifier

app = Flask(__name__)
app.secret_key = 'b630aeb22f15a8c5084a89a9b7126833b13906522c0620b8'

# Initialize the database
def init_db():
    conn = sqlite3.connect('voting.db')
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS users (
                      id INTEGER PRIMARY KEY,
                      username TEXT UNIQUE,
                      password TEXT)''')
    cursor.execute('''CREATE TABLE IF NOT EXISTS votes (
                      id INTEGER PRIMARY KEY,
                      user_id INTEGER,
                      vote TEXT,
                      timestamp TEXT,
                      FOREIGN KEY(user_id) REFERENCES users(id))''')
    conn.commit()
    conn.close()

init_db()

# Load face data and classifier
def load_data():
    names_path = os.path.join('data', 'names.pkl')
    faces_data_path = os.path.join('data', 'faces_data.pkl')

    if os.path.exists(names_path) and os.path.getsize(names_path) > 0:
        with open(names_path, 'rb') as f:
            LABELS = pickle.load(f)
    else:
        LABELS = []
        with open(names_path, 'wb') as f:
            pickle.dump(LABELS, f)

    if os.path.exists(faces_data_path) and os.path.getsize(faces_data_path) > 0:
        with open(faces_data_path, 'rb') as f:
            FACES = pickle.load(f)
    else:
        FACES = []
        with open(faces_data_path, 'wb') as f:
            pickle.dump(FACES, f)

    return LABELS, FACES

LABELS, FACES = load_data()

# Global KNN model
knn = None

# Function to train KNN Classifier
def train_knn():
    global knn
    if len(FACES) > 0 and len(LABELS) > 0:
        try:
            faces_data = np.array([face for face in FACES if isinstance(face, np.ndarray) and face.shape == (50*50*3,)])
            y = np.array(LABELS)

            if faces_data.size == 0 or y.size == 0:
                print("Face data or labels are empty.")
                return

            knn = KNeighborsClassifier(n_neighbors=3)
            knn.fit(faces_data, y)
            knn_model_path = os.path.join('models', 'knn_model.pkl')
            with open(knn_model_path, 'wb') as f:
                pickle.dump(knn, f)
        except Exception as e:
            print(f"Error training KNN: {e}")
    else:
        print("No face data to train KNN.")

# Load KNN Classifier
def load_knn_classifier():
    global knn
    knn_model_path = os.path.join('models', 'knn_model.pkl')
    if os.path.exists(knn_model_path):
        with open(knn_model_path, 'rb') as f:
            knn = pickle.load(f)
            print("KNN model loaded.")
    else:
        print("KNN model not found. It will be fitted later.")
    return knn

# Check if the face is registered
def is_face_registered(face):
    if knn is None:
        print("KNN is not fitted yet.")
        return False  # KNN is not fitted yet
    try:
        distances = knn.kneighbors([face], return_distance=True)
        return min(distances[0][0]) < 0.5  # Threshold can be adjusted
    except Exception as e:
        print(f"Error during distance calculation: {e}")
        return False

# Load KNN classifier on startup
load_knn_classifier()

# Load the face detection model (Haar Cascade Classifier)
facedetect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Homepage
@app.route('/')
def index():
    if 'user_id' in session:
        return render_template('index.html')
    return redirect(url_for('login'))

# Login route
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        conn = sqlite3.connect('voting.db')
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM users WHERE username=?', (username,))
        user = cursor.fetchone()
        conn.close()
        
        # Check if the user has already voted
        if user:
            conn = sqlite3.connect('voting.db')
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM votes WHERE user_id=?', (user[0],))
            voted = cursor.fetchone()
            conn.close()
            if voted:
                return 'You have already voted. Cannot log in with a different account.'

        if user and check_password_hash(user[2], password):
            session['user_id'] = user[0]
            session['username'] = username  # Store the username in session
            return redirect(url_for('index'))
        return 'Invalid username or password'
    return render_template('login.html')

# Register route
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = generate_password_hash(request.form['password'])
        conn = sqlite3.connect('voting.db')
        cursor = conn.cursor()
        try:
            cursor.execute('INSERT INTO users (username, password) VALUES (?, ?)', (username, password))
            conn.commit()
        except sqlite3.IntegrityError:
            conn.close()
            return 'Username already exists'
        conn.close()
        return redirect(url_for('login'))
    return render_template('register.html')

# Vote route
@app.route('/vote', methods=['POST'])
def vote():
    if 'user_id' in session:
        candidate = request.form.get('candidate')
        conn = sqlite3.connect('voting.db')
        cursor = conn.cursor()
        
        # Check if the user has already voted
        cursor.execute('SELECT * FROM votes WHERE user_id=?', (session['user_id'],))
        existing_vote = cursor.fetchone()
        
        if existing_vote:
            return jsonify({'status': 'You have already voted!'})

        # Record the vote
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute('INSERT INTO votes (user_id, vote, timestamp) VALUES (?, ?, ?)', (session['user_id'], candidate, ts))
        conn.commit()
        conn.close()
        return jsonify({'status': 'Vote recorded successfully!'})
    return jsonify({'status': 'User not authenticated!'})

# Results route
@app.route('/results')
def results():
    conn = sqlite3.connect('voting.db')
    cursor = conn.cursor()
    cursor.execute('SELECT vote, COUNT(*) FROM votes GROUP BY vote')
    results = cursor.fetchall()
    conn.close()
    return render_template('results.html', results=results)

# Video feed route
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Function to capture face data
@app.route('/capture', methods=['POST'])
def capture():
    if 'user_id' in session:
        # Check if the user has already voted
        conn = sqlite3.connect('voting.db')
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM votes WHERE user_id=?', (session['user_id'],))
        voted = cursor.fetchone()
        conn.close()
        
        if voted:
            return jsonify({'status': 'You have already voted. Cannot capture your face again.'})

        video = cv2.VideoCapture(0)
        if not video.isOpened():
            return jsonify({'status': 'Error opening video capture'})

        while True:
            ret, frame = video.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = facedetect.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                crop_img = frame[y:y + h, x:x + w]
                resized_img = cv2.resize(crop_img, (50, 50)).flatten()
                
                # Check if this face is already registered
                if is_face_registered(resized_img):
                    video.release()
                    return jsonify({'status': 'Face is already registered. Cannot capture again.'})

                # Add face data and label to training set
                FACES.append(resized_img)
                LABELS.append('Unknown')  # Use 'Unknown' as a placeholder

                # Update pickle files
                with open(os.path.join('data', 'faces_data.pkl'), 'wb') as f:
                    pickle.dump(FACES, f)
                with open(os.path.join('data', 'names.pkl'), 'wb') as f:
                    pickle.dump(LABELS, f)

                # Train KNN with new data
                train_knn()

            video.release()
            return jsonify({'status': 'Face captured successfully!'})

    return jsonify({'status': 'User not authenticated!'})

# Logout route
@app.route('/logout')
def logout():
    session.clear()  # Clear the session
    return redirect(url_for('login'))  # Redirect to login page

# Generate frames for video feed
def generate_frames():
    video = cv2.VideoCapture(0)
    while True:
        ret, frame = video.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facedetect.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            break

        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

if __name__ == '__main__':
    app.run(debug=True)
