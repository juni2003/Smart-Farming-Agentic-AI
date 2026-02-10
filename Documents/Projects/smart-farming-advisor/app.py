"""
Flask API Backend for Smart Farming Advisor
Exposes crop recommendation, disease detection, and Q&A endpoints
"""

import os
import sys
from pathlib import Path
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import traceback
from werkzeug.utils import secure_filename

# Load environment variables
load_dotenv()

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from src.agent.farming_agent import FarmingAgent
import config

# Flask app setup
app = Flask(__name__)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Upload folder
UPLOAD_FOLDER = Path(__file__).parent / "uploads"
UPLOAD_FOLDER.mkdir(exist_ok=True)
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'JPG', 'JPEG', 'PNG'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Initialize agent (will load on first request)
agent = None
agent_initialized = False


def init_agent():
    """Initialize farming agent once"""
    global agent, agent_initialized
    if not agent_initialized:
        print("\n🤖 Initializing Farming Agent...")
        agent = FarmingAgent()
        success = agent.initialize()
        agent_initialized = success
        return success
    return agent_initialized


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# ============================================================================
# HEALTH CHECK ENDPOINT
# ============================================================================

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'message': 'Smart Farming Advisor API is running',
        'agent_initialized': agent_initialized
    }), 200


# ============================================================================
# CROP RECOMMENDATION ENDPOINTS
# ============================================================================

@app.route('/api/crop/recommend', methods=['POST'])
def crop_recommend():
    """
    Crop recommendation endpoint
    
    Expected JSON:
    {
        "N": 90,
        "P": 42,
        "K": 43,
        "temperature": 20,
        "humidity": 82,
        "ph": 6.5,
        "rainfall": 202
    }
    """
    try:
        if not agent_initialized:
            init_agent()
        
        if not agent_initialized:
            return jsonify({
                'success': False,
                'error': 'Agent failed to initialize'
            }), 500
        
        data = request.get_json()
        
        # Validate required fields
        required = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
        missing = [k for k in required if k not in data]
        
        if missing:
            return jsonify({
                'success': False,
                'error': f'Missing fields: {", ".join(missing)}'
            }), 400
        
        # Call agent
        result = agent.process_query(
            "What crop should I plant?",
            crop_params={
                'N': float(data['N']),
                'P': float(data['P']),
                'K': float(data['K']),
                'temperature': float(data['temperature']),
                'humidity': float(data['humidity']),
                'ph': float(data['ph']),
                'rainfall': float(data['rainfall'])
            }
        )
        
        if 'error' in result:
            return jsonify({
                'success': False,
                'error': result['error']
            }), 400
        
        return jsonify({
            'success': True,
            'data': result['result'],
            'message': result['response']
        }), 200
        
    except Exception as e:
        print(f"Error in crop_recommend: {str(e)}")
        print(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/crop/validate', methods=['POST'])
def crop_validate():
    """Validate crop recommendation input parameters"""
    try:
        data = request.get_json()
        
        errors = {}
        
        # Validate ranges (based on typical farming conditions)
        validations = {
            'N': (0, 300, 'Nitrogen (kg/ha)'),
            'P': (0, 150, 'Phosphorus (kg/ha)'),
            'K': (0, 200, 'Potassium (kg/ha)'),
            'temperature': (-50, 60, 'Temperature (°C)'),
            'humidity': (0, 100, 'Humidity (%)'),
            'ph': (3, 10, 'Soil pH'),
            'rainfall': (0, 5000, 'Rainfall (mm)')
        }
        
        for field, (min_val, max_val, label) in validations.items():
            if field in data:
                val = float(data[field])
                if not (min_val <= val <= max_val):
                    errors[field] = f"{label} must be between {min_val} and {max_val}"
        
        return jsonify({
            'success': len(errors) == 0,
            'errors': errors
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# ============================================================================
# DISEASE DETECTION ENDPOINTS
# ============================================================================

@app.route('/api/disease/predict', methods=['POST'])
def disease_predict():
    """
    Disease detection endpoint
    
    Expects multipart form-data with 'file' field containing image
    """
    try:
        if not agent_initialized:
            init_agent()
        
        if not agent_initialized:
            return jsonify({
                'success': False,
                'error': 'Agent failed to initialize'
            }), 500
        
        # Check if file is in request
        if 'file' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No file provided'
            }), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No file selected'
            }), 400
        
        if not allowed_file(file.filename):
            return jsonify({
                'success': False,
                'error': f'File type not allowed. Allowed: {", ".join(ALLOWED_EXTENSIONS)}'
            }), 400
        
        # Save file temporarily
        filename = secure_filename(file.filename)
        filepath = UPLOAD_FOLDER / filename
        file.save(filepath)
        
        try:
            # Call agent
            result = agent.process_query(
                "My plant looks sick, can you identify the disease?",
                image_path=str(filepath)
            )
            
            if 'error' in result:
                return jsonify({
                    'success': False,
                    'error': result['error']
                }), 400
            
            return jsonify({
                'success': True,
                'data': result['result'],
                'message': result['response']
            }), 200
            
        finally:
            # Clean up uploaded file
            if filepath.exists():
                filepath.unlink()
        
    except Exception as e:
        print(f"Error in disease_predict: {str(e)}")
        print(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# ============================================================================
# FARMING Q&A ENDPOINTS
# ============================================================================

@app.route('/api/qa', methods=['POST'])
def farming_qa():
    """
    Farming Q&A endpoint
    
    Expected JSON:
    {
        "question": "What is the best time to plant rice?",
        "use_llm": false  // optional, default false
    }
    """
    try:
        if not agent_initialized:
            init_agent()
        
        if not agent_initialized:
            return jsonify({
                'success': False,
                'error': 'Agent failed to initialize'
            }), 500
        
        data = request.get_json()
        
        if 'question' not in data:
            return jsonify({
                'success': False,
                'error': 'Question field is required'
            }), 400
        
        question = data['question'].strip()
        
        if len(question) < 3:
            return jsonify({
                'success': False,
                'error': 'Question must be at least 3 characters long'
            }), 400
        
        use_llm = data.get('use_llm', False)
        
        # Call QA tool directly to avoid misclassification
        result = agent.qa_tool.answer_question(question, use_llm=use_llm, verbose=True)
        
        if 'error' in result:
            return jsonify({
                'success': False,
                'error': result['error']
            }), 400
        
        return jsonify({
            'success': True,
            'question': question,
            'answer': result.get('answer', 'No answer found.'),
            'retrieved_docs': result.get('retrieved_docs', []),
            'source': result.get('source', 'unknown')
        }), 200
        
    except Exception as e:
        print(f"Error in farming_qa: {str(e)}")
        print(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# ============================================================================
# SYSTEM INFO ENDPOINTS
# ============================================================================

@app.route('/api/system/info', methods=['GET'])
def system_info():
    """Get system information and model metrics"""
    try:
        import json
        
        info = {
            'system': 'Smart Farming Advisor',
            'version': '1.0.0',
            'components': {}
        }
        
        # Crop model info
        crop_metadata_path = config.MODELS_DIR / "crop_model_metadata.json"
        if crop_metadata_path.exists():
            with open(crop_metadata_path) as f:
                info['components']['crop_model'] = json.load(f)
        
        # Disease model info
        disease_metadata_path = config.MODELS_DIR / "disease_model_metadata.json"
        if disease_metadata_path.exists():
            with open(disease_metadata_path) as f:
                info['components']['disease_model'] = json.load(f)
        
        # FAQ stats
        faq_stats_path = config.PROCESSED_DATA_DIR / "faq_stats.json"
        if faq_stats_path.exists():
            with open(faq_stats_path) as f:
                info['components']['faq'] = json.load(f)
        
        return jsonify({
            'success': True,
            'data': info
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({
        'success': False,
        'error': 'Endpoint not found'
    }), 404


@app.errorhandler(500)
def server_error(error):
    """Handle 500 errors"""
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    print("\n" + "="*70)
    print("🚀 STARTING SMART FARMING ADVISOR API")
    print("="*70)
    
    # Initialize agent on startup
    print("\n📡 Initializing components on startup...")
    init_agent()
    
    print("\n✅ API server starting on http://localhost:5000")
    print("   - Health check: GET /health")
    print("   - Crop recommendation: POST /api/crop/recommend")
    print("   - Disease detection: POST /api/disease/predict")
    print("   - Farming Q&A: POST /api/qa")
    print("   - System info: GET /api/system/info")
    print("="*70 + "\n")
    
    # Run server
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,
        use_reloader=False  # Disable reloader to prevent agent re-initialization
    )
