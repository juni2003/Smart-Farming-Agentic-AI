"""
Crop Recommendation Tool
Uses trained Random Forest model to predict best crop
"""

import numpy as np
import joblib
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))
import config

class CropPredictorTool:
    """
    Tool for predicting crop recommendations
    """
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.is_loaded = False
        
    def load_model(self):
        """Load trained model and preprocessing objects"""
        print("📦 Loading Crop Prediction Model...")
        
        try:
            self.model = joblib.load(config.CROP_MODEL_PATH)
            self.scaler = joblib.load(config.CROP_SCALER_PATH)
            self.label_encoder = joblib.load(config.CROP_LABEL_ENCODER_PATH)
            self.is_loaded = True
            print("   ✅ Model loaded successfully")
            return True
        except Exception as e:
            print(f"   ❌ Error loading model: {e}")
            return False
    
    def create_features(self, N, P, K, temperature, humidity, ph, rainfall):
        """Create feature vector with engineered features"""
        
        # Original features
        features = [N, P, K, temperature, humidity, ph, rainfall]
        
        # Engineered features (same as preprocessing)
        soil_fertility = (N + P + K) / 3
        climate_index = temperature * humidity / 100
        npk_ratio = N / (P + K) if (P + K) > 0 else 0
        moisture_index = rainfall * humidity / 100
        
        # Add engineered features
        features.extend([soil_fertility, climate_index, npk_ratio, moisture_index])
        
        return np.array(features).reshape(1, -1)
    
    def predict(self, N, P, K, temperature, humidity, ph, rainfall, verbose=True):
        """
        Predict best crop for given conditions
        
        Args:
            N: Nitrogen content (kg/ha)
            P: Phosphorus content (kg/ha)
            K: Potassium content (kg/ha)
            temperature: Temperature (°C)
            humidity: Humidity (%)
            ph: Soil pH
            rainfall: Rainfall (mm)
            verbose: Print detailed output
            
        Returns:
            dict with prediction results
        """
        
        if not self.is_loaded:
            success = self.load_model()
            if not success:
                return {'error': 'Model not loaded'}
        
        try:
            # Create features
            features = self.create_features(N, P, K, temperature, humidity, ph, rainfall)
            
            # Scale features
            features_scaled = self.scaler.transform(features)
            
            # Predict
            prediction_encoded = self.model.predict(features_scaled)[0]
            predicted_crop = self.label_encoder.inverse_transform([prediction_encoded])[0]
            
            # Get probability scores
            probabilities = self.model.predict_proba(features_scaled)[0]
            confidence = float(probabilities[prediction_encoded])
            
            # Get top 3 recommendations
            top_3_indices = np.argsort(probabilities)[-3:][::-1]
            top_3_crops = self.label_encoder.inverse_transform(top_3_indices)
            top_3_probs = probabilities[top_3_indices]
            
            result = {
                'recommended_crop': predicted_crop,
                'confidence': confidence,
                'top_3_recommendations': [
                    {'crop': crop, 'probability': float(prob)}
                    for crop, prob in zip(top_3_crops, top_3_probs)
                ],
                'input_conditions': {
                    'N': N, 'P': P, 'K': K,
                    'temperature': temperature,
                    'humidity': humidity,
                    'ph': ph,
                    'rainfall': rainfall
                }
            }
            
            if verbose:
                print("\n" + "="*70)
                print("🌾 CROP RECOMMENDATION")
                print("="*70)
                print(f"\n📊 Input Conditions:")
                print(f"   Nitrogen (N):    {N} kg/ha")
                print(f"   Phosphorus (P):  {P} kg/ha")
                print(f"   Potassium (K):   {K} kg/ha")
                print(f"   Temperature:     {temperature}°C")
                print(f"   Humidity:        {humidity}%")
                print(f"   pH:              {ph}")
                print(f"   Rainfall:        {rainfall} mm")
                
                print(f"\n🏆 Recommended Crop: {predicted_crop}")
                print(f"   Confidence: {confidence*100:.2f}%")
                
                print(f"\n📋 Top 3 Recommendations:")
                for i, rec in enumerate(result['top_3_recommendations'], 1):
                    print(f"   {i}. {rec['crop']}: {rec['probability']*100:.2f}%")
                print("="*70)
            
            return result
            
        except Exception as e:
            return {'error': f'Prediction failed: {str(e)}'}

# =============================================================================
# TESTING
# =============================================================================

def test_crop_predictor():
    """Test crop predictor with sample inputs"""
    
    print("\n" + "="*70)
    print("🧪 TESTING CROP PREDICTOR TOOL")
    print("="*70)
    
    tool = CropPredictorTool()
    
    # Test cases
    test_cases = [
        {
            'name': 'Rice conditions',
            'N': 90, 'P': 42, 'K': 43,
            'temperature': 20, 'humidity': 82,
            'ph': 6.5, 'rainfall': 202
        },
        {
            'name': 'Wheat conditions',
            'N': 50, 'P': 30, 'K': 40,
            'temperature': 15, 'humidity': 60,
            'ph': 7.0, 'rainfall': 50
        },
        {
            'name': 'Cotton conditions',
            'N': 120, 'P': 60, 'K': 50,
            'temperature': 25, 'humidity': 70,
            'ph': 6.8, 'rainfall': 100
        }
    ]
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n{'='*70}")
        print(f"TEST CASE {i}: {test['name']}")
        print(f"{'='*70}")
        
        result = tool.predict(
            N=test['N'], P=test['P'], K=test['K'],
            temperature=test['temperature'],
            humidity=test['humidity'],
            ph=test['ph'],
            rainfall=test['rainfall'],
            verbose=True
        )
    
    print("\n" + "="*70)
    print("✅ CROP PREDICTOR TESTING COMPLETE!")
    print("="*70)

if __name__ == "__main__":
    test_crop_predictor()