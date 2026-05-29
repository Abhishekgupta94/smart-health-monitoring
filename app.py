from flask import Flask, render_template, request, redirect, url_for, session, flash, send_file, abort
from werkzeug.security import generate_password_hash, check_password_hash
import sqlite3
import pickle
import os
import re
import secrets
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
import io


# Load model & scaler ONCE (better performance)
model = None
scaler = None
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'heart_model.pkl')
SCALER_PATH = os.path.join(BASE_DIR, 'scaler.pkl')
DATABASE = os.path.join(BASE_DIR, 'database.db')

if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE='Lax',
)
if os.environ.get('FLASK_ENV') == 'production':
    app.config['SESSION_COOKIE_SECURE'] = True

USERNAME_PATTERN = re.compile(r'^[A-Za-z0-9_]{3,30}$')
EMAIL_PATTERN = re.compile(r'^[^@\s]+@[^@\s]+\.[^@\s]+$')

FIELD_RULES = {
    'age': (1, 120),
    'sex': {0, 1},
    'cp': {0, 1, 2, 3},
    'trestbps': (80, 250),
    'chol': (100, 700),
    'fbs': {0, 1},
    'restecg': {0, 1, 2},
    'thalachh': (60, 220),
    'exang': {0, 1},
    'oldpeak': (0.0, 10.0),
    'slope': {0, 1, 2},
    'ca': {0, 1, 2, 3, 4},
    'thal': {1, 2, 3},
}

DISPLAY_LABELS = {
    'sex': {
        0: 'Female',
        1: 'Male',
    },
    'cp': {
        0: 'Typical Angina',
        1: 'Atypical Angina',
        2: 'Non-anginal Pain',
        3: 'Asymptomatic',
    },
    'fbs': {
        0: 'No (≤120 mg/dl)',
        1: 'Yes (>120 mg/dl)',
    },
    'restecg': {
        0: 'Normal',
        1: 'ST-T Wave Abnormality',
        2: 'Left Ventricular Hypertrophy',
    },
    'exang': {
        0: 'No',
        1: 'Yes',
    },
    'slope': {
        0: 'Upsloping',
        1: 'Flat',
        2: 'Downsloping',
    },
    'thal': {
        1: 'Normal',
        2: 'Fixed Defect',
        3: 'Reversible Defect',
    },
}

def get_db_connection():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    conn.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            email TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    conn.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            age INTEGER,
            sex INTEGER,
            cp INTEGER,
            trestbps INTEGER,
            chol INTEGER,
            fbs INTEGER,
            restecg INTEGER,
            thalachh INTEGER,
            exang INTEGER,
            oldpeak REAL,
            slope INTEGER,
            ca INTEGER,
            thal INTEGER,
            probability REAL,
            risk_level TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (username) REFERENCES users (username)
        )
    ''')
    conn.commit()
    conn.close()


init_db()


def generate_csrf_token():
    token = session.get('csrf_token')
    if not token:
        token = secrets.token_hex(16)
        session['csrf_token'] = token
    return token


app.jinja_env.globals['csrf_token'] = generate_csrf_token


def validate_csrf():
    form_token = request.form.get('csrf_token', '')
    session_token = session.get('csrf_token', '')
    if not session_token or not secrets.compare_digest(form_token, session_token):
        abort(400, description='Invalid form submission. Please refresh the page and try again.')


def parse_numeric_field(field_name, cast_type):
    raw_value = request.form.get(field_name, '').strip()
    if raw_value == '':
        raise ValueError(f'{field_name} is required.')

    try:
        value = cast_type(raw_value)
    except ValueError as exc:
        raise ValueError(f'Invalid value for {field_name}.') from exc

    rule = FIELD_RULES[field_name]
    if isinstance(rule, tuple):
        min_value, max_value = rule
        if value < min_value or value > max_value:
            raise ValueError(f'{field_name} must be between {min_value} and {max_value}.')
    elif value not in rule:
        raise ValueError(f'Invalid option selected for {field_name}.')

    return value


def validate_registration_form(username, password, email):
    if not USERNAME_PATTERN.fullmatch(username):
        raise ValueError('Username must be 3-30 characters and use only letters, numbers, or underscores.')
    if len(password) < 8:
        raise ValueError('Password must be at least 8 characters long.')
    if email and not EMAIL_PATTERN.fullmatch(email):
        raise ValueError('Please enter a valid email address.')


def get_display_label(field_name, value):
    return DISPLAY_LABELS.get(field_name, {}).get(value, str(value))


def enrich_prediction(prediction):
    prediction_dict = dict(prediction)
    prediction_dict['sex_label'] = get_display_label('sex', prediction['sex'])
    prediction_dict['cp_label'] = get_display_label('cp', prediction['cp'])
    prediction_dict['fbs_label'] = get_display_label('fbs', prediction['fbs'])
    prediction_dict['restecg_label'] = get_display_label('restecg', prediction['restecg'])
    prediction_dict['exang_label'] = get_display_label('exang', prediction['exang'])
    prediction_dict['slope_label'] = get_display_label('slope', prediction['slope'])
    prediction_dict['thal_label'] = get_display_label('thal', prediction['thal'])
    return prediction_dict

@app.route('/')
def index():
    if 'username' in session:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        validate_csrf()
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        email = request.form.get('email', '').strip()

        try:
            validate_registration_form(username, password, email)
        except ValueError as exc:
            flash(str(exc), 'error')
            return redirect(url_for('register'))

        hashed_password = generate_password_hash(password)

        try:
            conn = get_db_connection()
            conn.execute('INSERT INTO users (username, password, email) VALUES (?, ?, ?)',
                        (username, hashed_password, email))
            conn.commit()
            conn.close()
            flash('Registration successful! Please login.', 'success')
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash('Username already exists!', 'error')
            return redirect(url_for('register'))

    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        validate_csrf()
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')

        if not username or not password:
            flash('Enter both username and password to sign in.', 'error')
            return redirect(url_for('login'))

        conn = get_db_connection()
        user = conn.execute('SELECT * FROM users WHERE username = ?', (username,)).fetchone()
        conn.close()

        if user and check_password_hash(user['password'], password):
            session['username'] = username
            flash('Login successful!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password!', 'error')

    return render_template('login.html')

@app.route('/dashboard')
def dashboard():
    if 'username' not in session:
        flash('Please login first!', 'error')
        return redirect(url_for('login'))

    return render_template('dashboard.html', username=session['username'])

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if 'username' not in session:
        flash('Please login first!', 'error')
        return redirect(url_for('login'))

    if request.method == 'POST':
            try:
                validate_csrf()
                age = parse_numeric_field('age', int)
                sex = parse_numeric_field('sex', int)
                cp = parse_numeric_field('cp', int)
                trestbps = parse_numeric_field('trestbps', int)
                chol = parse_numeric_field('chol', int)
                fbs = parse_numeric_field('fbs', int)
                restecg = parse_numeric_field('restecg', int)
                thalachh = parse_numeric_field('thalachh', int)
                exang = parse_numeric_field('exang', int)
                oldpeak = parse_numeric_field('oldpeak', float)
                slope = parse_numeric_field('slope', int)
                ca = parse_numeric_field('ca', int)
                thal = parse_numeric_field('thal', int)

                columns = ['age','sex','cp','trestbps','chol','fbs',
                        'restecg','thalachh','exang','oldpeak',
                        'slope','ca','thal']

                input_df = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs,
                                        restecg, thalachh, exang, oldpeak,
                                        slope, ca, thal]], columns=columns)

                if model is None or scaler is None:
                    flash("Model not found!", "error")
                    return redirect(url_for('predict'))

                input_scaled = scaler.transform(input_df)

                # The training dataset uses the positive label for the healthier class,
                # so invert the model output to get disease risk probability.
                healthy_probability = model.predict_proba(input_scaled)[0][1]
                probability = 1 - healthy_probability

                if probability < 0.4:
                    risk_level = 'Low Risk'
                elif probability < 0.7:
                    risk_level = 'Medium Risk'
                else:
                    risk_level = 'High Risk'

                # SAVE
                conn = get_db_connection()
                conn.execute('''
                    INSERT INTO predictions
                    (username, age, sex, cp, trestbps, chol, fbs, restecg,
                    thalachh, exang, oldpeak, slope, ca, thal, probability, risk_level)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (session['username'], age, sex, cp, trestbps, chol, fbs,
                    restecg, thalachh, exang, oldpeak, slope, ca, thal,
                    float(probability), risk_level))
                conn.commit()
                prediction_id = conn.execute('SELECT last_insert_rowid()').fetchone()[0]
                conn.close()

                return redirect(url_for('result', prediction_id=prediction_id)) 

            except Exception as e:
                flash(f'Unable to process the assessment: {str(e)}', 'error')
                return redirect(url_for('predict'))

    return render_template('predict.html')

@app.route('/result/<int:prediction_id>')
def result(prediction_id):
    if 'username' not in session:
        flash('Please login first!', 'error')
        return redirect(url_for('login'))

    conn = get_db_connection()
    prediction = conn.execute(
        'SELECT * FROM predictions WHERE id = ? AND username = ?',
        (prediction_id, session['username'])
    ).fetchone()
    conn.close()

    if not prediction:
        flash('Prediction not found!', 'error')
        return redirect(url_for('dashboard'))

    return render_template('result.html', prediction=enrich_prediction(prediction))

@app.route('/history')
def history():
    if 'username' not in session:
        flash('Please login first!', 'error')
        return redirect(url_for('login'))

    conn = get_db_connection()
    predictions = conn.execute(
        'SELECT * FROM predictions WHERE username = ? ORDER BY timestamp DESC',
        (session['username'],)
    ).fetchall()
    conn.close()

    return render_template('history.html', predictions=predictions)

@app.route('/download/<int:prediction_id>')
def download(prediction_id):
    if 'username' not in session:
        flash('Please login first!', 'error')
        return redirect(url_for('login'))

    conn = get_db_connection()
    prediction = conn.execute(
        'SELECT * FROM predictions WHERE id = ? AND username = ?',
        (prediction_id, session['username'])
    ).fetchone()
    conn.close()

    if not prediction:
        flash('Prediction not found!', 'error')
        return redirect(url_for('dashboard'))

    prediction = enrich_prediction(prediction)

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    elements = []
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#2c3e50'),
        spaceAfter=30,
        alignment=TA_CENTER
    )

    elements.append(Paragraph('Heart Disease Risk Assessment Summary', title_style))
    elements.append(Spacer(1, 0.3*inch))

    header_style = ParagraphStyle(
        'Header',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=colors.HexColor('#34495e'),
        spaceAfter=12
    )

    elements.append(Paragraph(f'Patient: {prediction["username"]}', header_style))
    elements.append(Paragraph(f'Date: {prediction["timestamp"]}', styles['Normal']))
    elements.append(Spacer(1, 0.3*inch))

    elements.append(Paragraph('Clinical Parameters', header_style))

    data = [
        ['Parameter', 'Value'],
        ['Age', str(prediction['age'])],
        ['Sex', prediction['sex_label']],
        ['Chest Pain Type', prediction['cp_label']],
        ['Resting Blood Pressure', f"{prediction['trestbps']} mm Hg"],
        ['Cholesterol', f"{prediction['chol']} mg/dl"],
        ['Fasting Blood Sugar', prediction['fbs_label']],
        ['Rest ECG', prediction['restecg_label']],
        ['Max Heart Rate', str(prediction['thalachh'])],
        ['Exercise Induced Angina', prediction['exang_label']],
        ['Oldpeak', str(prediction['oldpeak'])],
        ['Slope', prediction['slope_label']],
        ['CA', str(prediction['ca'])],
        ['Thalassemia', prediction['thal_label']],
    ]

    table = Table(data, colWidths=[3*inch, 3*inch])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3498db')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 10),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
    ]))

    elements.append(table)
    elements.append(Spacer(1, 0.3*inch))

    elements.append(Paragraph('Assessment Result', header_style))

    risk_color = colors.green if prediction['risk_level'] == 'Low Risk' else \
                 colors.orange if prediction['risk_level'] == 'Medium Risk' else colors.red

    result_data = [
        ['Risk Level', 'Probability'],
        [prediction['risk_level'], f"{prediction['probability']*100:.2f}%"],
    ]

    result_table = Table(result_data, colWidths=[3*inch, 3*inch])
    result_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2c3e50')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (0, 1), risk_color),
        ('TEXTCOLOR', (0, 1), (-1, 1), colors.white),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 1), (-1, -1), 14),
        ('GRID', (0, 0), (-1, -1), 2, colors.black),
        ('TOPPADDING', (0, 1), (-1, -1), 15),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 15),
    ]))

    elements.append(result_table)
    elements.append(Spacer(1, 0.3*inch))

    elements.append(Paragraph('General Recommendations', header_style))

    if prediction['risk_level'] == 'Low Risk':
        summary_title = 'Lower estimated risk'
        summary_text = (
            'Based on the submitted clinical values, this assessment indicates a lower '
            'estimated risk. Continue healthy habits and routine medical checkups.'
        )
        recommendations = [
            'Maintain a healthy lifestyle with regular exercise',
            'Continue a balanced diet with lower sodium and saturated fats',
            'Attend routine health checkups as recommended',
            'Keep monitoring blood pressure and cholesterol levels'
        ]
    elif prediction['risk_level'] == 'Medium Risk':
        summary_title = 'Follow-up recommended'
        summary_text = (
            'The submitted values suggest a moderate estimated risk. Monitor symptoms '
            'and discuss the result with a healthcare professional.'
        )
        recommendations = [
            'Monitor blood pressure and cholesterol regularly',
            'Discuss the result with a healthcare professional for further evaluation',
            'Adopt heart-healthy diet and exercise routine',
            'Consider stress management techniques',
            'Avoid smoking and limit alcohol consumption'
        ]
    else:
        summary_title = 'Prompt medical review advised'
        summary_text = (
            'The submitted values indicate a high estimated risk. Please seek timely '
            'medical advice for further evaluation.'
        )
        recommendations = [
            'Arrange a medical consultation as soon as possible',
            'A detailed cardiac evaluation may be required',
            'Lifestyle modifications are essential',
            'Regular monitoring of cardiac parameters',
            'Follow prescribed medication strictly',
            'Keep emergency and clinician contact details available'
        ]

    elements.append(Paragraph(summary_title, header_style))
    elements.append(Paragraph(summary_text, styles['Normal']))
    elements.append(Spacer(1, 0.2*inch))

    for rec in recommendations:
        elements.append(Paragraph(f'• {rec}', styles['Normal']))
        elements.append(Spacer(1, 0.1*inch))

    elements.append(Spacer(1, 0.3*inch))
    elements.append(Paragraph(
        'Medical disclaimer: This summary shows an estimated risk generated by a machine learning model. It should be used for educational and screening purposes only.',
        styles['Italic']
    ))

    doc.build(elements)
    buffer.seek(0)

    return send_file(
        buffer,
        as_attachment=True,
        download_name=f'heart_risk_report_{prediction_id}.pdf',
        mimetype='application/pdf'
    )

@app.route('/logout')
def logout():
    session.pop('username', None)
    session.pop('csrf_token', None)
    flash('You have been logged out successfully!', 'success')
    return redirect(url_for('login'))

if __name__ == '__main__':
    init_db()
    debug_mode = os.environ.get('FLASK_DEBUG', 'false').lower() in ('1', 'true', 'yes', 'on')
    port = int(os.environ.get('PORT', '5000'))
    app.run(debug=debug_mode, port=port)
