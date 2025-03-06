import sqlite3
import os

def init_db():
    # Ensure the data directory exists
    db_dir = os.path.join(os.path.dirname(__file__), 'data')
    if not os.path.exists(db_dir):
        os.makedirs(db_dir)
    
    # Define the database path
    db_path = os.path.join(db_dir, 'voting.db')
    
    # Connect to the SQLite database (it will create the file if it doesn't exist)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create tables
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

    cursor.execute('''CREATE TABLE IF NOT EXISTS faces (
                      id INTEGER PRIMARY KEY,
                      user_id INTEGER,
                      face_data BLOB,
                      FOREIGN KEY(user_id) REFERENCES users(id))''')  # New table for face data
    
    # Commit changes and close the connection
    conn.commit()
    conn.close()
    print(f"Database initialized and located at: {db_path}")

# Call the function to initialize the database
if __name__ == "__main__":
    init_db()
