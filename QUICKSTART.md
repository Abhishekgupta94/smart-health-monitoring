# Quick Start Guide

Get your Heart Disease Prediction System running in 5 minutes!

## Installation Steps

### 1. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 2. Create Sample ML Model Files (Optional)

If you don't have trained model files, create sample ones:

```bash
python create_sample_model.py
```

This will generate:
- `heart_model.pkl` - Sample prediction model
- `scaler.pkl` - Feature scaler

### 3. Run the Application

```bash
python app.py
```

You should see:
```
* Running on http://127.0.0.1:5000
```

### 4. Open in Browser

Navigate to: **http://localhost:5000**

## First Time Usage

### Step 1: Register
1. Click "Register here"
2. Enter username: `testuser`
3. Enter password: `password123`
4. Click "Register"

### Step 2: Login
1. Enter your username and password
2. Click "Login"

### Step 3: Start Prediction
1. Click "Begin Assessment" on dashboard
2. Fill in sample data:
   - Age: 55
   - Sex: Male
   - Chest Pain Type: Typical Angina
   - Resting BP: 130
   - Cholesterol: 250
   - Fasting Blood Sugar: No
   - Rest ECG: Normal
   - Max Heart Rate: 150
   - Exercise Angina: No
   - Oldpeak: 1.5
   - Slope: Flat
   - CA: 0
   - Thal: Normal
3. Click "Analyze Risk"

### Step 4: View Results
- See your risk assessment
- Click "Download PDF Report"
- View "History" to see all predictions

## Troubleshooting

**Error: Module not found**
```bash
pip install -r requirements.txt
```

**Error: Model files not found**
```bash
python create_sample_model.py
```

**Port already in use**
Edit `app.py` and change port 5000 to 5001

## Demo Accounts

You can create multiple accounts for testing:
- Username: `doctor1`, Password: `test123`
- Username: `patient1`, Password: `test123`

## Next Steps

- Read the full `README.md` for detailed documentation
- Train a real model with UCI Heart Disease dataset
- Customize the UI in `static/css/style.css`
- Add more features as needed

Enjoy your Heart Disease Prediction System!
