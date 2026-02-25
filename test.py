from dotenv import load_dotenv
import os
import psycopg2

# Load environment variables
load_dotenv()

# Get database URL
DATABASE_URL = os.getenv("DATABASE_URL")

# Connect
conn = psycopg2.connect(DATABASE_URL)

cursor = conn.cursor()

cursor.execute("SELECT COUNT(*) FROM prediction2;")
print(cursor.fetchone())