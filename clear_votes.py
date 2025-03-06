import sqlite3

def clear_votes():
    conn = sqlite3.connect('voting.db')
    cursor = conn.cursor()
    cursor.execute('DELETE FROM votes')
    conn.commit()
    conn.close()
    print("Votes cleared successfully.")

if __name__ == '__main__':
    clear_votes()
