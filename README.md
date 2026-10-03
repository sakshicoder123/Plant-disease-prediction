# 🌿 FloraScan AI - Intelligent Plant Disease Detection & Crop Health

An end-to-end Computer Vision & Deep Learning web application designed to detect, classify, and treat plant diseases across **39 distinct categories** and **14 agricultural crops**.

Built using **PyTorch**, **Convolutional Neural Networks (CNN)**, and **Flask**, trained on the **PlantVillage** dataset of over 61,000+ leaf images with **98%+ validation accuracy**.

---

## 🚀 Quick Start (One Command)

### 1. Requirements
Ensure Python 3.10+ or 3.11+ is installed.

### 2. Install Dependencies
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install flask pillow pandas numpy
```

### 3. Run Application
From the repository root, simply execute:
```bash
python run.py
```
Open your browser at **[http://127.0.0.1:5000](http://127.0.0.1:5000)**.

---

## 🌟 Key Features

1. **AI Diagnostic Suite (`/index` or `/diagnose`)**:
   - **Drag & Drop Upload**: Upload leaf photos with live preview.
   - **Device Camera Capture**: Take photos in real-time using your phone or webcam.
   - **One-Click Sample Gallery**: Test real sample leaves with a single click (Corn Rust, Apple Cedar Rust, Potato Blight, Healthy Tomato, etc.).
   
2. **Comprehensive Diagnostic Report (`/submit`)**:
   - Condition status (Healthy vs. Infected Pathogen).
   - Softmax **confidence score (%)**.
   - **Top-3 Alternate Differential Diagnoses** probability breakdown.
   - Detailed biological symptoms & prevention guidelines.
   - Targeted fertilizer / supplement recommendations with direct store links.

3. **Supplements & Fungicide Marketplace (`/market`)**:
   - Search products by crop or disease name.
   - Filter by treatments/fungicides vs. organic growth boosters.
   - Responsive multi-card product grid.

4. **Disease Encyclopedia (`/diseases`)**:
   - Reference guide for all 39 detectable conditions.
   - Filter by crop: Apple, Corn, Grape, Tomato, Potato, Pepper, etc.

5. **REST API Endpoint for Mobile & IoT (`/api/predict`)**:
   - `POST /api/predict` accepts multipart `image` uploads and returns JSON predictions with top-3 confidence scores and treatment steps.
   - `GET /api/status` returns health status of the neural model and device.

---

## 🍎 Supported Crops (14 Varieties)

| Crop | Detectable Conditions |
|---|---|
| **Apple** | Apple Scab, Black Rot, Cedar Apple Rust, Healthy |
| **Blueberry** | Healthy |
| **Cherry** | Powdery Mildew, Healthy |
| **Corn (Maize)** | Cercospora Leaf Spot, Common Rust, Northern Leaf Blight, Healthy |
| **Grape** | Black Rot, Esca (Black Measles), Leaf Blight (Isariopsis), Healthy |
| **Orange** | Huanglongbing (Citrus Greening) |
| **Peach** | Bacterial Spot, Healthy |
| **Pepper Bell** | Bacterial Spot, Healthy |
| **Potato** | Early Blight, Late Blight, Healthy |
| **Raspberry** | Healthy |
| **Soybean** | Healthy |
| **Squash** | Powdery Mildew |
| **Strawberry** | Leaf Scorch, Healthy |
| **Tomato** | Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Septoria Spot, Spider Mites, Target Spot, Mosaic Virus, Yellow Leaf Curl, Healthy |

---

## 🔬 Testing the Application

Run the automated test suite verifying all routes, neural inference, and REST endpoints:
```bash
python test_app.py
```
Expected output:
```text
[TEST] Running Comprehensive Test Suite for FloraScan AI...
[PASS] Test 1: GET / (Home Page) passed
[PASS] Test 2: GET /index (Diagnosis Page) passed
[PASS] Test 3: GET /market (Market Page) passed
[PASS] Test 4: GET /market?q=tomato (Search Filter) passed
[PASS] Test 5: GET /diseases (Encyclopedia) passed
[PASS] Test 6: GET /contact (Contact Page) passed
[PASS] Test 7: GET /api/status passed (Model loaded: True, Device: cpu)
[PASS] Test 8: GET /sample-diagnose/corn_common_rust.JPG passed
[PASS] Test 9: GET /sample-diagnose/apple_healthy.JPG passed
[PASS] Test 10: POST /api/predict passed -> Result: Potato (Early blight), Confidence: 90.8%
[SUCCESS] ALL 10 TESTS PASSED SUCCESSFULLY!
```
