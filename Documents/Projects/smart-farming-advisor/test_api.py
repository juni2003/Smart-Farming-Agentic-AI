#!/usr/bin/env python
"""
Quick test to verify backend is working and can accept requests
"""

import requests
import json

BASE_URL = "http://localhost:5000"

print("\n" + "="*70)
print("🧪 TESTING BACKEND CONNECTION")
print("="*70 + "\n")

# Test 1: Health Check
print("1️⃣  Testing Health Check...")
try:
    response = requests.get(f"{BASE_URL}/health")
    if response.status_code == 200:
        print(f"   ✅ Health check passed")
        print(f"   Response: {response.json()}")
    else:
        print(f"   ❌ Health check failed: {response.status_code}")
except Exception as e:
    print(f"   ❌ Error: {e}")

print()

# Test 2: Crop Recommendation
print("2️⃣  Testing Crop Recommendation...")
try:
    payload = {
        "N": 90,
        "P": 42,
        "K": 43,
        "temperature": 20,
        "humidity": 82,
        "ph": 6.5,
        "rainfall": 202
    }
    
    response = requests.post(f"{BASE_URL}/api/crop/recommend", json=payload)
    
    if response.status_code == 200:
        data = response.json()
        print(f"   ✅ Crop recommendation successful")
        print(f"   Recommended crop: {data['data']['recommended_crop']}")
        print(f"   Confidence: {data['data']['confidence']*100:.2f}%")
    else:
        print(f"   ❌ Request failed: {response.status_code}")
        print(f"   Error: {response.text}")
except Exception as e:
    print(f"   ❌ Error: {e}")

print()

# Test 3: Q&A
print("3️⃣  Testing Q&A...")
try:
    payload = {"question": "What is the best time to plant rice?"}
    
    response = requests.post(f"{BASE_URL}/api/qa", json=payload)
    
    if response.status_code == 200:
        data = response.json()
        print(f"   ✅ Q&A successful")
        print(f"   Question: {data['question']}")
        print(f"   Answer: {data['answer'][:100]}...")
    else:
        print(f"   ❌ Request failed: {response.status_code}")
        print(f"   Error: {response.text}")
except Exception as e:
    print(f"   ❌ Error: {e}")

print()

# Test 4: System Info
print("4️⃣  Testing System Info...")
try:
    response = requests.get(f"{BASE_URL}/api/system/info")
    
    if response.status_code == 200:
        data = response.json()
        print(f"   ✅ System info retrieved")
        if 'data' in data and 'components' in data['data']:
            components = data['data']['components']
            if 'crop_model' in components:
                print(f"   - Crop Model Accuracy: {components['crop_model'].get('accuracy', 'N/A')}")
            if 'disease_model' in components:
                print(f"   - Disease Model Accuracy: {components['disease_model'].get('test_accuracy', 'N/A')}")
    else:
        print(f"   ❌ Request failed: {response.status_code}")
except Exception as e:
    print(f"   ❌ Error: {e}")

print("\n" + "="*70)
print("✅ BACKEND CONNECTION TEST COMPLETE")
print("="*70 + "\n")
