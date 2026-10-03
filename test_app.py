import sys
import os
from pathlib import Path

# Ensure UTF-8 output
sys.stdout.reconfigure(encoding='utf-8')

# Add 'Flask Deployed App' to path
BASE_DIR = Path(__file__).resolve().parent / 'Flask Deployed App'
sys.path.insert(0, str(BASE_DIR))
os.chdir(str(BASE_DIR))

from app import app

def run_tests():
    print("[TEST] Running Comprehensive Test Suite for FloraScan AI...\n")
    client = app.test_client()

    # Test 1: Home Page
    r = client.get('/')
    assert r.status_code == 200, f"Home page failed: {r.status_code}"
    assert b"FloraScan AI" in r.data
    print("[PASS] Test 1: GET / (Home Page) passed")

    # Test 2: AI Engine / Diagnosis Page
    r = client.get('/index')
    assert r.status_code == 200, f"AI Engine page failed: {r.status_code}"
    assert b"Diagnostic Suite" in r.data
    print("[PASS] Test 2: GET /index (Diagnosis Page) passed")

    # Test 3: Supplements Market
    r = client.get('/market')
    assert r.status_code == 200, f"Market page failed: {r.status_code}"
    assert b"Supplements" in r.data
    print("[PASS] Test 3: GET /market (Market Page) passed")

    # Test 4: Market Search
    r = client.get('/market?q=tomato')
    assert r.status_code == 200, f"Market search failed: {r.status_code}"
    print("[PASS] Test 4: GET /market?q=tomato (Search Filter) passed")

    # Test 5: Disease Encyclopedia
    r = client.get('/diseases')
    assert r.status_code == 200, f"Diseases page failed: {r.status_code}"
    assert b"Plant Disease Encyclopedia" in r.data
    print("[PASS] Test 5: GET /diseases (Encyclopedia) passed")

    # Test 6: Contact Page
    r = client.get('/contact')
    assert r.status_code == 200, f"Contact page failed: {r.status_code}"
    assert b"Get in Touch" in r.data
    print("[PASS] Test 6: GET /contact (Contact Page) passed")

    # Test 7: API Status
    r = client.get('/api/status')
    assert r.status_code == 200, f"API Status failed: {r.status_code}"
    data = r.get_json()
    assert data['status'] == 'online'
    assert data['model_loaded'] is True
    print(f"[PASS] Test 7: GET /api/status passed (Model loaded: {data['model_loaded']}, Device: {data['device']})")

    # Test 8: Sample Diagnose (Corn Common Rust)
    r = client.get('/sample-diagnose/corn_common_rust.JPG')
    assert r.status_code == 200, f"Sample diagnose failed: {r.status_code}"
    assert b"Corn" in r.data
    assert b"Common rust" in r.data or b"Rust" in r.data
    print("[PASS] Test 8: GET /sample-diagnose/corn_common_rust.JPG passed")

    # Test 9: Sample Diagnose (Healthy Apple)
    r = client.get('/sample-diagnose/apple_healthy.JPG')
    assert r.status_code == 200, f"Healthy apple test failed: {r.status_code}"
    assert b"Healthy" in r.data
    print("[PASS] Test 9: GET /sample-diagnose/apple_healthy.JPG (Healthy status) passed")

    # Test 10: POST /api/predict with image file
    sample_img_path = BASE_DIR / 'static' / 'samples' / 'potato_early_blight.JPG'
    with open(sample_img_path, 'rb') as f:
        r = client.post('/api/predict', data={'image': (f, 'test_potato.jpg')}, content_type='multipart/form-data')
    assert r.status_code == 200, f"API predict failed: {r.status_code}"
    res = r.get_json()
    assert res['success'] is True
    assert res['result']['crop'] == 'Potato'
    assert 'Early' in res['result']['condition']
    print(f"[PASS] Test 10: POST /api/predict passed -> Result: {res['result']['crop']} ({res['result']['condition']}), Confidence: {res['result']['confidence']}%")

    print("\n[SUCCESS] ALL 10 TESTS PASSED SUCCESSFULLY! The application and AI model are 100% functional!")

if __name__ == '__main__':
    run_tests()
