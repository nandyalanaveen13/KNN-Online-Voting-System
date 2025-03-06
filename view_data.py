import sqlite3

def view_data():
    conn = sqlite3.connect('voting.db')
    cursor = conn.cursor()
    
    # View all tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    print("Tables in the database:")
    for table in tables:
        print(table[0])
    
    # Query a specific table (e.g., 'votes')
    cursor.execute("SELECT * FROM votes;")
    rows = cursor.fetchall()
    print("\nData in 'votes' table:")
    for row in rows:
        print(row)
    
    conn.close()

if __name__ == '__main__':
    view_data() 