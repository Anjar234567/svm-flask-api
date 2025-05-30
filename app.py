from flask import Flask, request, jsonify
import joblib
import numpy as np
from PIL import Image
from skimage.feature import hog
import traceback
import os
import requests

app = Flask(__name__)

# 🔹 Fungsi untuk download file dari Google Drive
def download_file_from_google_drive(file_id, dest_path):
    if not os.path.exists(dest_path):
        print(f"⬇️ Downloading {dest_path} from Google Drive...")
        url = f"https://drive.google.com/uc?export=download&id={file_id}"
        r = requests.get(url)
        with open(dest_path, 'wb') as f:
            f.write(r.content)
        print(f"✅ Downloaded: {dest_path}")
    else:
        print(f"📁 {dest_path} already exists.")

# 🔹 Download otomatis model dan scaler
download_file_from_google_drive("1ynM1X10ac-jvD16AHzdJE-nL5ASfxA60", "model_svm.pkl")
download_file_from_google_drive("1fG5PqD2vhHh41KeSS7uZqNSTvUFAqrON", "scaler.pkl")

# 🔹 Load model dan scaler
model = joblib.load('model_svm.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/')
def home():
    return '✅ Flask aktif dan menjalankan file app.py ini!'

# 🔹 Route klasifikasi gambar
@app.route('/predict', methods=['POST'])
def predict():
    try:
        print("🚀 Menerima request ke /predict")

        file = request.files['image']
        img = Image.open(file.stream).convert('L')
        img = img.resize((256, 256))
        img_np = np.array(img)

        features = hog(img_np, pixels_per_cell=(16, 16),
                       cells_per_block=(2, 2), orientations=9)

        features_scaled = scaler.transform([features])
        hasil = model.predict(features_scaled)[0]

        # Info tambahan
        info_tanaman = {
            "jahe": {
                "nama_latin": "Zingiber officinale",
                "manfaat": "Meningkatkan daya tahan tubuh, menghangatkan badan, mengatasi mual dan nyeri otot."
            },
            "kunyit": {
                "nama_latin": "Curcuma longa",
                "manfaat": "Anti-inflamasi, menjaga kesehatan pencernaan, membantu mengatasi masalah kulit."
            },
            "temulawak": {
                "nama_latin": "Curcuma xanthorrhiza",
                "manfaat": "Menjaga kesehatan hati, meningkatkan nafsu makan, dan antioksidan alami."
            }
        }

        detail = info_tanaman.get(hasil, {"nama_latin": "-", "manfaat": "-"})

        return jsonify({
            'hasil': hasil,
            'nama_latin': detail["nama_latin"],
            'manfaat': detail["manfaat"]
        })

    except Exception as e:
        print("❌ Terjadi error:", e)
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

# 🔹 Jalankan Flask
if __name__ == '__main__':
    print("🚀 Menjalankan server Flask...")
    app.run(host='0.0.0.0', port=5000)
