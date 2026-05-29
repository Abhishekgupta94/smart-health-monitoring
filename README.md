# Smart Health Monitoring

Smart Health Monitoring is a Flask-based web application that estimates heart disease risk from clinical inputs using a trained machine learning model. The system includes user authentication, prediction history, downloadable PDF reports, and a responsive dashboard for running and reviewing assessments.

## Project Summary

This project was built as an educational health-tech application that combines:

- a Flask web backend
- a machine learning prediction pipeline
- SQLite-based local data storage
- a Bootstrap-based user interface
- PDF report generation for saved assessments

The current version uses a trained model generated from `cleaned_merged_heart_dataset.csv` through `train_model.py`.

## Features

- User registration and login with hashed passwords
- Heart disease risk assessment using 13 clinical parameters
- Three-level classification: Low Risk, Medium Risk, High Risk
- Assessment history for each logged-in user
- PDF report download for every saved result
- Heart health tips on the dashboard
- Server-side validation for authentication and prediction input
- CSRF protection for submitted forms

## Tech Stack

- Backend: Flask
- Frontend: HTML, CSS, Bootstrap 5
- Database: SQLite
- Machine Learning: pandas, scikit-learn, numpy
- PDF Generation: ReportLab
- Authentication: Werkzeug password hashing

## Project Structure

```text
Smart Health Monitoring/
├── app.py
├── train_model.py
├── create_sample_model.py
├── requirements.txt
├── heart_model.pkl
├── scaler.pkl
├── database.db
├── templates/
│   ├── login.html
│   ├── register.html
│   ├── dashboard.html
│   ├── predict.html
│   ├── result.html
│   └── history.html
└── static/
    └── css/
        └── style.css
```

## Dataset and Model

The model is trained using the dataset:

- `cleaned_merged_heart_dataset.csv`

Required feature columns:

- `age`
- `sex`
- `cp`
- `trestbps`
- `chol`
- `fbs`
- `restecg`
- `thalachh`
- `exang`
- `oldpeak`
- `slope`
- `ca`
- `thal`

Target column:

- `target`

Training is handled by `train_model.py`, which:

1. loads the dataset
2. validates the required columns
3. splits the data into train and test sets
4. scales the features with `StandardScaler`
5. trains a `RandomForestClassifier`
6. prints evaluation metrics
7. saves `heart_model.pkl` and `scaler.pkl`

## Setup Instructions

### 1. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the model

Place `cleaned_merged_heart_dataset.csv` either in the project folder or its parent folder, then run:

```bash
python train_model.py
```

This creates:

- `heart_model.pkl`
- `scaler.pkl`

### 4. Run the application

Windows PowerShell example:

```powershell
$env:SECRET_KEY="replace-with-a-strong-secret-key"
$env:FLASK_DEBUG="false"
$env:PORT="5000"
python app.py
```

Open the application at:

- `http://127.0.0.1:5000`

## How to Use the Application

1. Create a user account on the registration page.
2. Sign in with your username and password.
3. Open `New Assessment` from the dashboard.
4. Enter all 13 clinical input values.
5. Submit the form to generate a risk assessment.
6. Review the result and recommendations.
7. Download the PDF report or open the saved entry in history.

## Risk Classification Logic

- Low Risk: probability below `0.40`
- Medium Risk: probability from `0.40` to below `0.70`
- High Risk: probability `0.70` or higher

## Database Schema

### `users`

- `id`
- `username`
- `password`
- `email`
- `created_at`

### `predictions`

- `id`
- `username`
- `age`
- `sex`
- `cp`
- `trestbps`
- `chol`
- `fbs`
- `restecg`
- `thalachh`
- `exang`
- `oldpeak`
- `slope`
- `ca`
- `thal`
- `probability`
- `risk_level`
- `timestamp`

## Security and Validation

The current implementation includes:

- password hashing using Werkzeug
- parameterized SQLite queries
- session-based authentication
- CSRF token validation on forms
- server-side validation for username, email, password, and prediction values
- secure session cookie defaults (`HttpOnly`, `SameSite=Lax`)

## Troubleshooting

### Model files not found

Run:

```bash
python train_model.py
```

### Database issue

Delete `database.db` and restart the app. The tables will be recreated automatically.

### Port already in use

Set a different port before running:

```powershell
$env:PORT="5001"
python app.py
```

### Dependency issue

Upgrade installer tools and reinstall:

```bash
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

## Limitations

- This system is for educational and screening purposes only.
- It does not replace professional diagnosis or treatment.
- Predictions depend on dataset quality and model training assumptions.
- The application stores data locally in SQLite and is not production-hardened for clinical deployment.

## Future Improvements

- Admin dashboard for clinicians
- Better explanation of feature meanings in reports
- Charts for long-term patient history
- Email or alert notifications for high-risk outcomes
- Deployment with a production database and HTTPS

## License

This project is intended for academic and educational use.
