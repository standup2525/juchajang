from flask import Flask, request, jsonify, render_template
import sqlite3
from datetime import datetime, timedelta
import os

app = Flask(__name__)

# Database setup
def init_db():
    conn = sqlite3.connect('parking.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS parking_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            plate_number TEXT NOT NULL,
            entry_time TEXT NOT NULL,
            exit_time TEXT,
            status TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

def cleanup_old_records():
    """Remove records older than 1 month"""
    conn = sqlite3.connect('parking.db')
    c = conn.cursor()
    one_month_ago = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d %H:%M:%S')
    c.execute('DELETE FROM parking_records WHERE entry_time < ?', (one_month_ago,))
    conn.commit()
    conn.close()

# API endpoints
@app.route('/api/parking', methods=['POST'])
def handle_parking():
    data = request.json
    plate_number = data.get('plate_number')
    timestamp = data.get('timestamp')
    
    if not plate_number or not timestamp:
        return jsonify({'error': 'Missing required data'}), 400
    
    conn = sqlite3.connect('parking.db')
    c = conn.cursor()
    
    # Check if vehicle is already parked
    c.execute('SELECT id, status FROM parking_records WHERE plate_number = ? AND status = "parked"',
              (plate_number,))
    existing = c.fetchone()
    
    if existing:
        # Update exit time
        c.execute('UPDATE parking_records SET exit_time = ?, status = "exited" WHERE id = ?',
                  (timestamp, existing[0]))
        message = f"Vehicle {plate_number} has exited"
    else:
        # Create new parking record
        c.execute('INSERT INTO parking_records (plate_number, entry_time, status) VALUES (?, ?, "parked")',
                  (plate_number, timestamp))
        message = f"Vehicle {plate_number} has entered"
    
    conn.commit()
    conn.close()
    
    return jsonify({'message': message}), 200

# Web interface
@app.route('/')
def index():
    conn = sqlite3.connect('parking.db')
    c = conn.cursor()
    
    # Get current parking status
    c.execute('''
        SELECT plate_number, entry_time, exit_time, status
        FROM parking_records
        ORDER BY entry_time DESC
        LIMIT 100
    ''')
    records = c.fetchall()
    
    conn.close()
    
    return render_template('index.html', records=records)

# Create templates directory and index.html
os.makedirs('templates', exist_ok=True)
with open('templates/index.html', 'w') as f:
    f.write('''
<!DOCTYPE html>
<html>
<head>
    <title>Parking System</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f0f0f0;
        }
        h1 {
            color: #333;
            text-align: center;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
            background-color: white;
            box-shadow: 0 1px 3px rgba(0,0,0,0.2);
        }
        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        th {
            background-color: #4CAF50;
            color: white;
        }
        tr:nth-child(even) {
            background-color: #f2f2f2;
        }
        tr:hover {
            background-color: #ddd;
        }
        .status-parked {
            color: #4CAF50;
            font-weight: bold;
        }
        .status-exited {
            color: #666;
        }
    </style>
</head>
<body>
    <h1>Parking System Status</h1>
    <table>
        <tr>
            <th>Plate Number</th>
            <th>Entry Time</th>
            <th>Exit Time</th>
            <th>Status</th>
        </tr>
        {% for record in records %}
        <tr>
            <td>{{ record[0] }}</td>
            <td>{{ record[1] }}</td>
            <td>{{ record[2] if record[2] else '-' }}</td>
            <td class="status-{{ record[3] }}">{{ record[3] }}</td>
        </tr>
        {% endfor %}
    </table>
</body>
</html>
    ''')

if __name__ == '__main__':
    init_db()
    cleanup_old_records()
    app.run(host='0.0.0.0', port=5000) 