import os
import shutil
import sqlite3
import csv
import numpy as np
from datetime import datetime
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import pickle
import matplotlib
matplotlib.use('Agg')  # Non-GUI backend
import matplotlib.pyplot as plt
import pandas as pd
from gtts import gTTS

app = Flask(__name__)

# ------------------ Folders ------------------
os.makedirs('static/images', exist_ok=True)
os.makedirs('static/audio', exist_ok=True)
os.makedirs('upload', exist_ok=True)

DB_FILE = 'user_data.db'
HISTORY_FILE = 'detection_history.csv'
MODEL_PATH = 'currency_classification.h5'
CLASS_PATH = 'class_names.pkl'

# ------------------ Load model ------------------
model = load_model(MODEL_PATH)
with open(CLASS_PATH, 'rb') as f:
    class_names = pickle.load(f)

# ------------------ Currency reasons ------------------
reasons = {
    '10_fake': "The ₹10 note is fake due to mismatched watermark or incorrect texture pattern.",
    '20_fake': "The ₹20 note is fake because of irregular color tone and missing security thread.",
    '50_fake': "The ₹50 note is fake — holographic strip and microtext mismatch.",
    '100_fake': "The ₹100 note is fake — Gandhi portrait and color contrast deviate.",
    '200_fake': "The ₹200 note is fake — security thread alignment inconsistent.",
    '500_fake': "The ₹500 note is fake — optical ink differs.",
    '2000_fake': "The ₹2000 note is fake — print quality not authentic.",
    '10_real': "The ₹10 note is real — watermark, texture, and serial number correct.",
    '20_real': "The ₹20 note is real — color tone, micro-lettering, verified.",
    '50_real': "The ₹50 note is real — hologram and print quality correct.",
    '100_real': "The ₹100 note is real — Gandhi portrait clarity correct.",
    '200_real': "The ₹200 note is real — watermark alignment correct.",
    '500_real': "The ₹500 note is real — optical ink verified.",
    '2000_real': "The ₹2000 note is real — watermark, optical ink, microtext correct."
}

# ------------------ Helper functions ------------------
def append_to_history(filename, predicted_class, confidence, status, reason):
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    header = ['Timestamp', 'Filename', 'Prediction', 'Confidence', 'Status', 'Reason']
    exists = os.path.exists(HISTORY_FILE)
    with open(HISTORY_FILE, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow(header)
        writer.writerow([now, filename, predicted_class, f"{confidence:.2f}", status, reason])

def generate_audio(text, lang='en'):
    filename = f"{datetime.now().strftime('%Y%m%d%H%M%S')}.mp3"
    audio_path = os.path.join('static/audio', filename)
    gTTS(text=text, lang=lang).save(audio_path)
    return f"/static/audio/{filename}"

def preprocess_image(img_path):
    img = load_img(img_path, target_size=(150, 150))
    img_array = img_to_array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

def predict_image(img_path):
    img = preprocess_image(img_path)
    preds = model.predict(img)
    idx = np.argmax(preds)
    return class_names[idx], float(preds[0][idx])

def generate_bar_graph():
    if not os.path.exists(HISTORY_FILE):
        return None
    df = pd.read_csv(HISTORY_FILE, on_bad_lines='skip')
    df['Denomination'] = df['Prediction'].apply(lambda x: ''.join([c for c in x if c.isdigit()]))
    df['Category'] = df['Prediction'].apply(lambda x: 'Real' if 'real' in str(x).lower() else 'Fake')
    summary = df.groupby(['Denomination', 'Category']).size().unstack(fill_value=0).sort_index()
    
    summary.plot(kind='bar', stacked=True, color=['green', 'red'], figsize=(8,5))
    plt.title("Currency Detection Summary by Denomination")
    plt.xlabel("Denomination (₹)")
    plt.ylabel("Number of Notes")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    graph_path = 'static/detection_summary.png'
    plt.savefig(graph_path)
    plt.close()
    return graph_path

# ------------------ Routes ------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/userreg', methods=['POST'])
def userreg():
    name = request.form['name']
    password = request.form['password']
    mobile = request.form['phone']
    email = request.form['email']
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS user(name TEXT, password TEXT, mobile TEXT, email TEXT)""")
    c.execute("INSERT INTO user VALUES (?, ?, ?, ?)", (name, password, mobile, email))
    conn.commit()
    conn.close()
    return render_template('index.html', msg='User registered successfully!')

@app.route('/userlog', methods=['POST'])
def userlog():
    name = request.form['name']
    password = request.form['password']
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT * FROM user WHERE name=? AND password=?", (name, password))
    user = c.fetchone()
    conn.close()
    if user:
        return render_template('userlog.html')
    else:
        return render_template('index.html', msg='Invalid credentials')

@app.route('/image', methods=['GET' ,'POST'])
def image():
    file = request.files['file']
    if not file:
        return "No file uploaded", 400
    filename = file.filename
    upload_path = os.path.join('upload', filename)
    file.save(upload_path)
    dst_path = os.path.join('static/images', filename)
    shutil.copy(upload_path, dst_path)

    predicted_class, confidence = predict_image(dst_path)
    denomination = ''.join([c for c in predicted_class if c.isdigit()])
    note_type = 'real' if 'real' in predicted_class.lower() else 'fake'
    reason = reasons.get(f"{denomination}_{note_type}", "Unable to determine note authenticity.")
    status = 'Real' if note_type == 'real' else 'Fake'

    # ✅ Voice output includes reason now
    announce_text = (
        f"This note is genuine. {reason}" if status == 'Real'
        else f"This note is counterfeit. {reason}"
    )
    audio_url = generate_audio(announce_text, lang='en')

    append_to_history(filename, predicted_class, confidence, status, reason)
    graph_path = generate_bar_graph()

    return render_template('results.html',
                           status=status,
                           reason=reason,
                           accuracy=f"{confidence*100:.2f}%",
                           image=f"/static/images/{filename}",
                           audio=audio_url,
                           graph=f"/{graph_path}" if graph_path else None)

@app.route('/play_audio', methods=['POST'])
def play_audio():
    status = request.form['status']
    reason = request.form['reason']
    accuracy = request.form['accuracy']
    lang = request.form['lang']

    announcements = {
        'Real': {
            'kn': "ಈ ನೋಟು ನಿಜವಾಗಿದೆ.",
            'en': "This note is genuine.",
            'hi': "यह नोट असली है।",
            'ta': "இந்த நோட்டு உண்மையானது.",
            'ml': "ഈ നോട്ട് യഥാർത്ഥമാണ്."
        },
        'Fake': {
            'kn': "ಈ ನೋಟು ನಕಲಿ ಆಗಿದೆ.",
            'en': "This note is counterfeit.",
            'hi': "यह नोट नकली है।",
            'ta': "இந்த நோட்டு போலியாக உள்ளது.",
            'ml': "ഈ നോട്ട് പണിയില്ലാത്തതാണ്."
        }
    }

    announce_text = announcements[status].get(lang, announcements[status]['en'])
    
    # ✅ Include reason in voice output
    full_text = f"{announce_text} {reason}"
    audio_url = generate_audio(full_text, lang)

    return render_template('results.html',
                           status=status,
                           reason=reason,
                           accuracy=accuracy,
                           image=request.form.get('image', ''),
                           audio=audio_url,
                           graph=request.form.get('graph', None))

@app.route('/graph')
def graph():
    graph_path = generate_bar_graph()
    if not graph_path:
        return render_template('graph.html', msg="No data yet.")
    return render_template('graph.html',
                           graph=f"/{graph_path}",
                           msg="Currency Detection Summary")

@app.route('/history')
def history():
    try:
        df = pd.read_csv(HISTORY_FILE, on_bad_lines='skip')
        records = df.to_dict(orient='records')
        print(records)
    except Exception:
        records = []
        print(records)
    return render_template('history.html', records=records)

@app.route('/logout')
def logout():
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
