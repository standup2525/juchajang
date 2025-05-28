#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
차량 번호판 인식 시스템 데모용 웹서버
Flask를 사용하여 차량 정보를 수신하고 관리하는 웹서버
"""

from flask import Flask, request, jsonify, render_template, send_from_directory
import os
import json
import sqlite3
from datetime import datetime
import logging
from werkzeug.utils import secure_filename

# ================================
# 웹서버 설정
# ================================

# Flask 앱 설정
app = Flask(__name__)
app.config['SECRET_KEY'] = 'vehicle_detection_demo_key'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB 최대 파일 크기

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('web_server.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 업로드 디렉토리 생성
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ================================
# 데이터베이스 설정
# ================================

def init_database():
    """데이터베이스 초기화"""
    conn = sqlite3.connect('vehicle_records.db')
    cursor = conn.cursor()
    
    # 차량 기록 테이블 생성
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS vehicle_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            plate_number TEXT NOT NULL,
            direction TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            device_id TEXT,
            confidence REAL,
            image_path TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # 시스템 통계 테이블 생성
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS system_stats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            total_detections INTEGER DEFAULT 0,
            entry_count INTEGER DEFAULT 0,
            exit_count INTEGER DEFAULT 0,
            unique_vehicles INTEGER DEFAULT 0,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    conn.commit()
    conn.close()
    logger.info("데이터베이스 초기화 완료")

def save_vehicle_record(plate_number, direction, timestamp, device_id=None, confidence=None, image_path=None):
    """차량 기록을 데이터베이스에 저장"""
    try:
        conn = sqlite3.connect('vehicle_records.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO vehicle_records 
            (plate_number, direction, timestamp, device_id, confidence, image_path)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (plate_number, direction, timestamp, device_id, confidence, image_path))
        
        conn.commit()
        record_id = cursor.lastrowid
        conn.close()
        
        # 일일 통계 업데이트
        update_daily_stats(direction)
        
        logger.info(f"차량 기록 저장 완료: {plate_number} ({direction})")
        return record_id
        
    except Exception as e:
        logger.error(f"차량 기록 저장 실패: {e}")
        return None

def update_daily_stats(direction):
    """일일 통계 업데이트"""
    try:
        today = datetime.now().strftime('%Y-%m-%d')
        conn = sqlite3.connect('vehicle_records.db')
        cursor = conn.cursor()
        
        # 오늘 날짜의 통계 확인
        cursor.execute('SELECT id FROM system_stats WHERE date = ?', (today,))
        existing = cursor.fetchone()
        
        if existing:
            # 기존 통계 업데이트
            if direction == 'entry':
                cursor.execute('''
                    UPDATE system_stats 
                    SET total_detections = total_detections + 1,
                        entry_count = entry_count + 1
                    WHERE date = ?
                ''', (today,))
            elif direction == 'exit':
                cursor.execute('''
                    UPDATE system_stats 
                    SET total_detections = total_detections + 1,
                        exit_count = exit_count + 1
                    WHERE date = ?
                ''', (today,))
        else:
            # 새로운 통계 생성
            entry_count = 1 if direction == 'entry' else 0
            exit_count = 1 if direction == 'exit' else 0
            cursor.execute('''
                INSERT INTO system_stats (date, total_detections, entry_count, exit_count)
                VALUES (?, 1, ?, ?)
            ''', (today, entry_count, exit_count))
        
        conn.commit()
        conn.close()
        
    except Exception as e:
        logger.error(f"통계 업데이트 실패: {e}")

def get_recent_records(limit=50):
    """최근 차량 기록 조회"""
    try:
        conn = sqlite3.connect('vehicle_records.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT plate_number, direction, timestamp, device_id, confidence, image_path, created_at
            FROM vehicle_records 
            ORDER BY created_at DESC 
            LIMIT ?
        ''', (limit,))
        
        records = cursor.fetchall()
        conn.close()
        
        return [
            {
                'plate_number': record[0],
                'direction': record[1],
                'timestamp': record[2],
                'device_id': record[3],
                'confidence': record[4],
                'image_path': record[5],
                'created_at': record[6]
            }
            for record in records
        ]
        
    except Exception as e:
        logger.error(f"기록 조회 실패: {e}")
        return []

def get_daily_stats():
    """일일 통계 조회"""
    try:
        conn = sqlite3.connect('vehicle_records.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT date, total_detections, entry_count, exit_count
            FROM system_stats 
            ORDER BY date DESC 
            LIMIT 7
        ''')
        
        stats = cursor.fetchall()
        conn.close()
        
        return [
            {
                'date': stat[0],
                'total_detections': stat[1],
                'entry_count': stat[2],
                'exit_count': stat[3]
            }
            for stat in stats
        ]
        
    except Exception as e:
        logger.error(f"통계 조회 실패: {e}")
        return []

# ================================
# API 엔드포인트
# ================================

@app.route('/api/vehicle', methods=['POST'])
def receive_vehicle_data():
    """차량 정보 수신 API"""
    try:
        # JSON 데이터 파싱
        data = request.form.to_dict()
        
        # 필수 필드 확인
        required_fields = ['plate_number', 'direction', 'timestamp']
        for field in required_fields:
            if field not in data:
                return jsonify({
                    'success': False,
                    'error': f'필수 필드 누락: {field}'
                }), 400
        
        plate_number = data['plate_number']
        direction = data['direction']
        timestamp = data['timestamp']
        device_id = data.get('device_id', 'unknown')
        confidence = float(data.get('confidence', 0.0)) if data.get('confidence') else None
        
        # 이미지 파일 처리
        image_path = None
        if 'image' in request.files:
            file = request.files['image']
            if file and file.filename:
                filename = secure_filename(f"{plate_number}_{direction}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
                image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(image_path)
                logger.info(f"이미지 저장 완료: {image_path}")
        
        # 데이터베이스에 저장
        record_id = save_vehicle_record(
            plate_number, direction, timestamp, device_id, confidence, image_path
        )
        
        if record_id:
            logger.info(f"차량 정보 수신 성공: {plate_number} ({direction})")
            return jsonify({
                'success': True,
                'message': '차량 정보가 성공적으로 저장되었습니다.',
                'record_id': record_id
            }), 200
        else:
            return jsonify({
                'success': False,
                'error': '데이터베이스 저장 실패'
            }), 500
            
    except Exception as e:
        logger.error(f"차량 정보 수신 실패: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/records', methods=['GET'])
def get_records():
    """차량 기록 조회 API"""
    try:
        limit = int(request.args.get('limit', 50))
        records = get_recent_records(limit)
        
        return jsonify({
            'success': True,
            'records': records,
            'count': len(records)
        }), 200
        
    except Exception as e:
        logger.error(f"기록 조회 실패: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """통계 조회 API"""
    try:
        daily_stats = get_daily_stats()
        
        # 오늘 통계
        today = datetime.now().strftime('%Y-%m-%d')
        today_stats = next((stat for stat in daily_stats if stat['date'] == today), {
            'date': today,
            'total_detections': 0,
            'entry_count': 0,
            'exit_count': 0
        })
        
        return jsonify({
            'success': True,
            'today': today_stats,
            'daily_stats': daily_stats
        }), 200
        
    except Exception as e:
        logger.error(f"통계 조회 실패: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# ================================
# 웹 페이지 라우트
# ================================

@app.route('/')
def index():
    """메인 대시보드 페이지"""
    return render_template('dashboard.html')

@app.route('/records')
def records_page():
    """차량 기록 페이지"""
    return render_template('records.html')

@app.route('/stats')
def stats_page():
    """통계 페이지"""
    return render_template('stats.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """업로드된 이미지 파일 서빙"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# ================================
# 템플릿 생성
# ================================

def create_templates():
    """HTML 템플릿 파일들 생성"""
    templates_dir = 'templates'
    os.makedirs(templates_dir, exist_ok=True)
    
    # 기본 레이아웃 템플릿
    base_template = '''<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{% block title %}차량 번호판 인식 시스템{% endblock %}</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <style>
        .navbar-brand { font-weight: bold; }
        .card { box-shadow: 0 0.125rem 0.25rem rgba(0, 0, 0, 0.075); }
        .direction-entry { color: #28a745; }
        .direction-exit { color: #dc3545; }
        .stats-card { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; }
    </style>
</head>
<body>
    <nav class="navbar navbar-expand-lg navbar-dark bg-primary">
        <div class="container">
            <a class="navbar-brand" href="/">
                <i class="fas fa-car"></i> 차량 번호판 인식 시스템
            </a>
            <div class="navbar-nav ms-auto">
                <a class="nav-link" href="/">대시보드</a>
                <a class="nav-link" href="/records">차량 기록</a>
                <a class="nav-link" href="/stats">통계</a>
            </div>
        </div>
    </nav>
    
    <div class="container mt-4">
        {% block content %}{% endblock %}
    </div>
    
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
    {% block scripts %}{% endblock %}
</body>
</html>'''
    
    # 대시보드 템플릿
    dashboard_template = '''{% extends "base.html" %}

{% block title %}대시보드 - 차량 번호판 인식 시스템{% endblock %}

{% block content %}
<div class="row">
    <div class="col-md-12">
        <h1 class="mb-4">
            <i class="fas fa-tachometer-alt"></i> 실시간 대시보드
        </h1>
    </div>
</div>

<div class="row mb-4">
    <div class="col-md-3">
        <div class="card stats-card">
            <div class="card-body text-center">
                <h5 class="card-title">오늘 총 감지</h5>
                <h2 id="total-today">-</h2>
                <i class="fas fa-car fa-2x"></i>
            </div>
        </div>
    </div>
    <div class="col-md-3">
        <div class="card bg-success text-white">
            <div class="card-body text-center">
                <h5 class="card-title">입차</h5>
                <h2 id="entry-today">-</h2>
                <i class="fas fa-sign-in-alt fa-2x"></i>
            </div>
        </div>
    </div>
    <div class="col-md-3">
        <div class="card bg-danger text-white">
            <div class="card-body text-center">
                <h5 class="card-title">출차</h5>
                <h2 id="exit-today">-</h2>
                <i class="fas fa-sign-out-alt fa-2x"></i>
            </div>
        </div>
    </div>
    <div class="col-md-3">
        <div class="card bg-info text-white">
            <div class="card-body text-center">
                <h5 class="card-title">현재 주차</h5>
                <h2 id="current-parked">-</h2>
                <i class="fas fa-parking fa-2x"></i>
            </div>
        </div>
    </div>
</div>

<div class="row">
    <div class="col-md-8">
        <div class="card">
            <div class="card-header">
                <h5><i class="fas fa-list"></i> 최근 차량 기록</h5>
            </div>
            <div class="card-body">
                <div class="table-responsive">
                    <table class="table table-striped" id="recent-records">
                        <thead>
                            <tr>
                                <th>번호판</th>
                                <th>방향</th>
                                <th>시간</th>
                                <th>이미지</th>
                            </tr>
                        </thead>
                        <tbody>
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    </div>
    <div class="col-md-4">
        <div class="card">
            <div class="card-header">
                <h5><i class="fas fa-chart-line"></i> 주간 통계</h5>
            </div>
            <div class="card-body">
                <canvas id="weekly-chart" width="400" height="200"></canvas>
            </div>
        </div>
    </div>
</div>
{% endblock %}

{% block scripts %}
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<script>
    function loadDashboard() {
        // 통계 로드
        $.get('/api/stats', function(data) {
            if (data.success) {
                $('#total-today').text(data.today.total_detections);
                $('#entry-today').text(data.today.entry_count);
                $('#exit-today').text(data.today.exit_count);
                $('#current-parked').text(data.today.entry_count - data.today.exit_count);
                
                // 주간 차트 업데이트
                updateWeeklyChart(data.daily_stats);
            }
        });
        
        // 최근 기록 로드
        $.get('/api/records?limit=10', function(data) {
            if (data.success) {
                updateRecentRecords(data.records);
            }
        });
    }
    
    function updateRecentRecords(records) {
        const tbody = $('#recent-records tbody');
        tbody.empty();
        
        records.forEach(function(record) {
            const directionClass = record.direction === 'entry' ? 'direction-entry' : 'direction-exit';
            const directionIcon = record.direction === 'entry' ? 'fa-sign-in-alt' : 'fa-sign-out-alt';
            const directionText = record.direction === 'entry' ? '입차' : '출차';
            
            const imageCell = record.image_path ? 
                `<img src="/uploads/${record.image_path.split('/').pop()}" class="img-thumbnail" style="max-width: 50px;">` : 
                '-';
            
            const row = `
                <tr>
                    <td><strong>${record.plate_number}</strong></td>
                    <td><span class="${directionClass}"><i class="fas ${directionIcon}"></i> ${directionText}</span></td>
                    <td>${new Date(record.created_at).toLocaleString('ko-KR')}</td>
                    <td>${imageCell}</td>
                </tr>
            `;
            tbody.append(row);
        });
    }
    
    function updateWeeklyChart(stats) {
        const ctx = document.getElementById('weekly-chart').getContext('2d');
        
        const labels = stats.map(stat => stat.date).reverse();
        const entryData = stats.map(stat => stat.entry_count).reverse();
        const exitData = stats.map(stat => stat.exit_count).reverse();
        
        new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [{
                    label: '입차',
                    data: entryData,
                    borderColor: '#28a745',
                    backgroundColor: 'rgba(40, 167, 69, 0.1)',
                    tension: 0.1
                }, {
                    label: '출차',
                    data: exitData,
                    borderColor: '#dc3545',
                    backgroundColor: 'rgba(220, 53, 69, 0.1)',
                    tension: 0.1
                }]
            },
            options: {
                responsive: true,
                scales: {
                    y: {
                        beginAtZero: true
                    }
                }
            }
        });
    }
    
    // 페이지 로드 시 대시보드 로드
    $(document).ready(function() {
        loadDashboard();
        
        // 30초마다 자동 새로고침
        setInterval(loadDashboard, 30000);
    });
</script>
{% endblock %}'''
    
    # 파일 저장
    with open(os.path.join(templates_dir, 'base.html'), 'w', encoding='utf-8') as f:
        f.write(base_template)
    
    with open(os.path.join(templates_dir, 'dashboard.html'), 'w', encoding='utf-8') as f:
        f.write(dashboard_template)
    
    logger.info("HTML 템플릿 파일 생성 완료")

# ================================
# 메인 실행부
# ================================

if __name__ == '__main__':
    # 데이터베이스 초기화
    init_database()
    
    # 템플릿 생성
    create_templates()
    
    logger.info("웹서버 시작 중...")
    logger.info("대시보드 URL: http://localhost:5000")
    logger.info("API 엔드포인트: http://localhost:5000/api/vehicle")
    
    # 개발 서버 실행
    app.run(host='0.0.0.0', port=5000, debug=True) 