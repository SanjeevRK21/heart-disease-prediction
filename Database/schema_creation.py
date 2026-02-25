import sqlite3

conn = sqlite3.connect("database2.db")
cursor = conn.cursor()

# Create the correct table
cursor.execute('''CREATE TABLE IF NOT EXISTS prediction2 (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    age INTEGER,
    sex INTEGER,
    chestpaintype INTEGER,
    restingbp INTEGER,
    cholesterol INTEGER,
    fastingbs INTEGER,
    restingecg INTEGER,
    maxhr INTEGER,
    exerciseangina INTEGER,
    oldpeak REAL,
    st_slope INTEGER,
    prediction TEXT,  -- Changed to TEXT
    probability REAL  -- Changed to REAL
)''')

conn.commit()
conn.close()

print("Table 'prediction2' created successfully!")
