
Final year Project
 **KNN-based Online Voting System Using Machine Learning**:  

## Overview  
This project is a **secure, efficient, and user-friendly online voting system** that leverages **machine learning** and **biometric authentication** to ensure a fraud-proof election process. It integrates **K-Nearest Neighbors (KNN) face recognition** with traditional username-password authentication to verify voters, preventing duplicate or unauthorized voting.

## ✨ Features  
- 🔐 **Secure Authentication**: Users log in using a **username, password**, and **face recognition**.  
- 🎭 **Real-Time Face Recognition**: Uses **OpenCV and KNN** to authenticate voters.  
- 📜 **One Vote per User**: Prevents multiple votes from a single person.  
- 📊 **Live Vote Counting**: Displays election results dynamically.  
- 📡 **Web-Based Platform**: Accessible from any device with a webcam.  

## 🔧 Technology Stack  
- **Frontend**: HTML, CSS, JavaScript  
- **Backend**: Flask (Python)  
- **Database**: SQLite  
- **Machine Learning**: KNN (scikit-learn)  
- **Face Recognition**: OpenCV  

## 🚀 Installation & Setup  
1. **Clone the repository**  
   ```bash
   git clone https://github.com/your-repo/online-voting-knn.git
   cd online-voting-knn
   ```
2. **Install dependencies**  
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the Flask app**  
   ```bash
   python app.py
   ```
4. **Access the system** via `http://127.0.0.1:5000/`  

## 📌 How It Works  
1. **Register** with a username, password, and facial data.  
2. **Log in** using credentials and verify identity via **face recognition**.  
3. **Cast your vote** securely.  
4. **View real-time election results**.  


