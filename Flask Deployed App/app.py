import os
import re
import uuid
from pathlib import Path
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

import CNN

# Base directory for relative paths
BASE_DIR = Path(__file__).resolve().parent

# Setup Flask application
app = Flask(
    __name__,
    template_folder=str(BASE_DIR / 'templates'),
    static_folder=str(BASE_DIR / 'static')
)
app.secret_key = os.environ.get('SECRET_KEY', 'plant-disease-detection-secret-key-2026')

UPLOAD_FOLDER = BASE_DIR / 'static' / 'uploads'
SAMPLES_FOLDER = BASE_DIR / 'static' / 'samples'
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
app.config['UPLOAD_FOLDER'] = str(UPLOAD_FOLDER)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max upload
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp', 'JPG', 'PNG', 'JPEG'}

# Load metadata CSV files
disease_info_path = BASE_DIR / 'disease_info.csv'
supplement_info_path = BASE_DIR / 'supplement_info.csv'

disease_info = pd.read_csv(disease_info_path, encoding='cp1252')
supplement_info = pd.read_csv(supplement_info_path, encoding='cp1252')

# Setup Model and Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CNN.CNN(39)
model_path = BASE_DIR / 'plant_disease_model_1_latest.pt'

if model_path.exists():
    model.load_state_dict(torch.load(str(model_path), map_location=device))
    print(f"Loaded model successfully from {model_path} on {device}")
else:
    print(f"WARNING: Model file {model_path} not found!")

model.to(device)
model.eval()

# Supported crops metadata
CROPS_META = [
    {"name": "Apple", "icon": "🍎", "diseases": 3, "healthy": True},
    {"name": "Blueberry", "icon": "🫐", "diseases": 0, "healthy": True},
    {"name": "Cherry", "icon": "🍒", "diseases": 1, "healthy": True},
    {"name": "Corn", "icon": "🌽", "diseases": 3, "healthy": True},
    {"name": "Grape", "icon": "🍇", "diseases": 3, "healthy": True},
    {"name": "Orange", "icon": "🍊", "diseases": 1, "healthy": False},
    {"name": "Peach", "icon": "🍑", "diseases": 1, "healthy": True},
    {"name": "Pepper Bell", "icon": "🫑", "diseases": 1, "healthy": True},
    {"name": "Potato", "icon": "🥔", "diseases": 2, "healthy": True},
    {"name": "Raspberry", "icon": "🍓", "diseases": 0, "healthy": True},
    {"name": "Soybean", "icon": "🌱", "diseases": 0, "healthy": True},
    {"name": "Squash", "icon": "🎃", "diseases": 1, "healthy": False},
    {"name": "Strawberry", "icon": "🍓", "diseases": 1, "healthy": True},
    {"name": "Tomato", "icon": "🍅", "diseases": 9, "healthy": True},
]

# Curated sample images for instant one-click testing
SAMPLE_TEST_IMAGES = [
    {"file": "Apple_ceder_apple_rust.JPG", "name": "Apple: Cedar Rust", "crop": "Apple"},
    {"file": "apple_healthy.JPG", "name": "Apple: Healthy Leaf", "crop": "Apple"},
    {"file": "corn_common_rust.JPG", "name": "Corn: Common Rust", "crop": "Corn"},
    {"file": "corn_healthy.jpg", "name": "Corn: Healthy Leaf", "crop": "Corn"},
    {"file": "Grape_esca.JPG", "name": "Grape: Esca Measles", "crop": "Grape"},
    {"file": "potato_early_blight.JPG", "name": "Potato: Early Blight", "crop": "Potato"},
    {"file": "potato_healthy.JPG", "name": "Potato: Healthy Leaf", "crop": "Potato"},
    {"file": "tomato_bacterial_spot.JPG", "name": "Tomato: Bacterial Spot", "crop": "Tomato"},
    {"file": "tomato_early_blight.JPG", "name": "Tomato: Early Blight", "crop": "Tomato"},
    {"file": "tomato_healthy.JPG", "name": "Tomato: Healthy Leaf", "crop": "Tomato"},
    {"file": "background_without_leaves.jpg", "name": "Non-Leaf Background", "crop": "Other"},
]

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in [ext.lower() for ext in ALLOWED_EXTENSIONS]

def clean_class_name(raw_name):
    """Turns 'Tomato___Early_blight' into ('Tomato', 'Early Blight')"""
    if raw_name == 'Background_without_leaves':
        return 'Background', 'No Leaf Detected'
    parts = raw_name.split('___')
    crop = parts[0].replace('_', ' ').strip()
    disease = parts[1].replace('_', ' ').strip() if len(parts) > 1 else 'Unknown'
    return crop, disease

def run_prediction_on_image(image_input):
    """
    Takes an image file path or PIL Image object and runs inference.
    Returns structured prediction details.
    """
    if isinstance(image_input, (str, Path)):
        img = Image.open(str(image_input)).convert('RGB')
    else:
        img = image_input.convert('RGB')

    img_resized = img.resize((224, 224))
    tensor = TF.to_tensor(img_resized).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1)[0]
        top_probs, top_indices = torch.topk(probs, 3)

    pred_idx = top_indices[0].item()
    confidence = top_probs[0].item() * 100.0

    raw_class = CNN.idx_to_classes.get(pred_idx, 'Unknown')
    crop_name, condition_name = clean_class_name(raw_class)

    # Health determination
    is_healthy = 'healthy' in raw_class.lower()
    is_background = raw_class == 'Background_without_leaves'

    # Alternate predictions
    alternates = []
    for p, idx in zip(top_probs, top_indices):
        c_raw = CNN.idx_to_classes.get(idx.item(), 'Unknown')
        c_crop, c_cond = clean_class_name(c_raw)
        alternates.append({
            'index': idx.item(),
            'raw_name': c_raw,
            'crop': c_crop,
            'condition': c_cond,
            'confidence': round(p.item() * 100.0, 2)
        })

    # Pull metadata from CSVs safely
    title = disease_info['disease_name'].iloc[pred_idx] if pred_idx < len(disease_info) else raw_class
    description = disease_info['description'].iloc[pred_idx] if pred_idx < len(disease_info) else "Information not available."
    prevent = disease_info['Possible Steps'].iloc[pred_idx] if pred_idx < len(disease_info) else "Maintain standard crop management."
    image_url = disease_info['image_url'].iloc[pred_idx] if pred_idx < len(disease_info) else ""

    supplement_name = supplement_info['supplement name'].iloc[pred_idx] if pred_idx < len(supplement_info) else "Recommended Plant Food"
    supplement_image = supplement_info['supplement image'].iloc[pred_idx] if pred_idx < len(supplement_info) else ""
    supplement_buy = supplement_info['buy link'].iloc[pred_idx] if pred_idx < len(supplement_info) else "#"

    return {
        'pred_idx': pred_idx,
        'confidence': round(confidence, 1),
        'crop': crop_name,
        'condition': condition_name,
        'title': title,
        'description': description,
        'prevent': prevent,
        'image_url': image_url,
        'supplement_name': supplement_name,
        'supplement_image': supplement_image,
        'supplement_buy': supplement_buy,
        'is_healthy': is_healthy,
        'is_background': is_background,
        'alternates': alternates
    }

# --- ROUTES ---

@app.route('/')
def home():
    return render_template(
        'home.html',
        crops=CROPS_META,
        samples=SAMPLE_TEST_IMAGES[:6]
    )

@app.route('/index')
@app.route('/diagnose')
def ai_engine_page():
    return render_template(
        'index.html',
        samples=SAMPLE_TEST_IMAGES
    )

@app.route('/submit', methods=['GET', 'POST'])
def submit():
    if request.method == 'GET':
        return redirect(url_for('ai_engine_page'))

    # Check if a file was uploaded
    if 'image' not in request.files or request.files['image'].filename == '':
        flash('Please select or capture a leaf photo first.', 'warning')
        return redirect(url_for('ai_engine_page'))

    file = request.files['image']
    if not allowed_file(file.filename):
        flash('Allowed image formats are JPG, JPEG, PNG, WEBP.', 'danger')
        return redirect(url_for('ai_engine_page'))

    # Save securely with unique ID to avoid overwriting or caching issues
    original_ext = file.filename.rsplit('.', 1)[1].lower()
    unique_filename = f"leaf_{uuid.uuid4().hex[:10]}.{original_ext}"
    saved_path = UPLOAD_FOLDER / unique_filename
    file.save(str(saved_path))

    # Run AI inference
    result = run_prediction_on_image(saved_path)
    relative_img_url = url_for('static', filename=f'uploads/{unique_filename}')

    return render_template(
        'submit.html',
        **result,
        user_image=relative_img_url,
        pred=result['pred_idx'],
        desc=result['description'],
        sname=result['supplement_name'],
        simage=result['supplement_image'],
        buy_link=result['supplement_buy']
    )

@app.route('/sample-diagnose/<filename>')
def sample_diagnose(filename):
    """Allows one-click diagnostic testing using included sample leaf images"""
    sample_path = SAMPLES_FOLDER / filename
    if not sample_path.exists():
        # Fallback to test_images folder at repo root if needed
        root_test_path = BASE_DIR.parent / 'test_images' / filename
        if root_test_path.exists():
            sample_path = root_test_path
        else:
            flash(f"Sample leaf '{filename}' not found.", 'warning')
            return redirect(url_for('ai_engine_page'))

    result = run_prediction_on_image(sample_path)
    relative_img_url = url_for('static', filename=f'samples/{filename}')

    return render_template(
        'submit.html',
        **result,
        user_image=relative_img_url,
        pred=result['pred_idx'],
        desc=result['description'],
        sname=result['supplement_name'],
        simage=result['supplement_image'],
        buy_link=result['supplement_buy']
    )

@app.route('/market')
def market():
    query = request.args.get('q', '').strip().lower()
    category = request.args.get('cat', 'all').lower()

    items = []
    for idx in range(len(supplement_info)):
        if idx == 4:  # Background without leaves
            continue
        
        disease_name = str(disease_info['disease_name'].iloc[idx])
        supp_name = str(supplement_info['supplement name'].iloc[idx])
        supp_img = str(supplement_info['supplement image'].iloc[idx])
        buy_link = str(supplement_info['buy link'].iloc[idx])
        is_healthy = idx in [3, 5, 7, 11, 15, 18, 20, 23, 24, 25, 28, 38]

        # Filter by search
        if query and (query not in disease_name.lower() and query not in supp_name.lower()):
            continue
        # Filter by category
        if category == 'healthy' and not is_healthy:
            continue
        if category == 'treatment' and is_healthy:
            continue

        items.append({
            'index': idx,
            'disease': disease_name,
            'name': supp_name,
            'image': supp_img,
            'buy': buy_link,
            'is_healthy': is_healthy
        })

    return render_template(
        'market.html',
        items=items,
        query=query,
        category=category,
        total_count=len(items)
    )

@app.route('/diseases')
def diseases_catalog():
    """Encyclopedia page listing all 39 detectable conditions"""
    catalog = []
    for idx in range(len(disease_info)):
        if idx == 4:  # skip background
            continue
        raw_class = CNN.idx_to_classes.get(idx, 'Unknown')
        crop, condition = clean_class_name(raw_class)
        is_healthy = idx in [3, 5, 7, 11, 15, 18, 20, 23, 24, 25, 28, 38]
        catalog.append({
            'index': idx,
            'raw_class': raw_class,
            'crop': crop,
            'condition': condition,
            'name': disease_info['disease_name'].iloc[idx],
            'description': disease_info['description'].iloc[idx],
            'prevent': disease_info['Possible Steps'].iloc[idx],
            'image_url': disease_info['image_url'].iloc[idx],
            'is_healthy': is_healthy
        })

    # Group by crop
    crops_grouped = {}
    for item in catalog:
        c = item['crop']
        if c not in crops_grouped:
            crops_grouped[c] = []
        crops_grouped[c].append(item)

    return render_template('diseases.html', catalog=catalog, crops_grouped=crops_grouped)

@app.route('/contact')
def contact():
    return render_template('contact-us.html')

# --- JSON REST API for Mobile / IoT integrations ---

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """REST API endpoint for mobile/IoT clients"""
    if 'image' not in request.files or request.files['image'].filename == '':
        return jsonify({'success': False, 'error': 'No image file provided'}), 400

    file = request.files['image']
    if not allowed_file(file.filename):
        return jsonify({'success': False, 'error': 'Invalid image extension'}), 400

    try:
        img = Image.open(file.stream).convert('RGB')
        result = run_prediction_on_image(img)
        return jsonify({
            'success': True,
            'result': result
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/status', methods=['GET'])
def api_status():
    return jsonify({
        'status': 'online',
        'classes_count': 39,
        'device': str(device),
        'model_loaded': model_path.exists()
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"\n* Starting Plant Disease Detection System on http://127.0.0.1:{port}")
    app.run(host='0.0.0.0', port=port, debug=True)
