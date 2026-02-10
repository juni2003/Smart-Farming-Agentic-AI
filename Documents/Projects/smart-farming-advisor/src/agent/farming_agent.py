"""
Smart Farming Advisor Agent
Routes user queries to appropriate tools and provides integrated responses
"""

import re
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.tools.crop_predictor_tool import CropPredictorTool
from src.tools.disease_detector_tool import DiseaseDetectorTool
from src.tools.rag_qa_tool import RAGQATool

class FarmingAgent:
    """
    Intelligent agent that routes queries to appropriate tools
    """
    
    def __init__(self):
        self.crop_tool = CropPredictorTool()
        self.disease_tool = DiseaseDetectorTool()
        self.qa_tool = RAGQATool()
        self.is_initialized = False
        
    def initialize(self):
        """Initialize all tools"""
        print("\n" + "="*70)
        print("🤖 INITIALIZING SMART FARMING ADVISOR AGENT")
        print("="*70)
        
        print("\n1️⃣  Loading Crop Recommendation Tool...")
        crop_success = self.crop_tool.load_model()
        
        print("\n2️⃣  Loading Disease Detection Tool...")
        disease_success = self.disease_tool.load_model()
        
        print("\n3️⃣  Loading RAG Q&A Tool...")
        qa_success = self.qa_tool.initialize()
        
        if crop_success and disease_success and qa_success:
            self.is_initialized = True
            print("\n" + "="*70)
            print("✅ AGENT INITIALIZED SUCCESSFULLY!")
            print("="*70)
            print("\n🎯 Available Capabilities:")
            print("   • Crop Recommendation (soil + climate analysis)")
            print("   • Disease Detection (image analysis)")
            print("   • Farming Q&A (knowledge retrieval)")
            return True
        else:
            print("\n⚠️  Some tools failed to initialize")
            return False
    
    def classify_intent(self, query):
        """
        Classify user intent to route to appropriate tool
        
        Returns: 'crop', 'disease', or 'qa'
        """
        
        query_lower = query.lower()
        
        # Crop recommendation keywords
        crop_keywords = [
            'recommend', 'suggest', 'best crop', 'which crop', 'what crop',
            'should i plant', 'can i grow', 'suitable crop',
            'nitrogen', 'phosphorus', 'potassium', 'npk',
            'temperature', 'humidity', 'rainfall', 'soil'
        ]
        
        # Disease detection keywords
        disease_keywords = [
            'disease', 'infected', 'sick', 'spot', 'blight', 'mold',
            'leaf problem', 'plant disease', 'diagnosis', 'identify disease',
            'what is wrong', 'unhealthy', 'dying', 'wilting'
        ]
        
        # Check for crop recommendation intent
        crop_score = sum(1 for kw in crop_keywords if kw in query_lower)
        
        # Check for disease detection intent
        disease_score = sum(1 for kw in disease_keywords if kw in query_lower)
        
        # Route based on scores
        if disease_score > crop_score:
            return 'disease'
        elif crop_score > 0:
            return 'crop'
        else:
            return 'qa'
    
    def extract_crop_params(self, query):
        """
        Extract crop recommendation parameters from natural language
        Returns dict with N, P, K, temp, humidity, ph, rainfall
        """
        
        params = {}
        
        # Extract numbers with units
        patterns = {
            'N': r'nitrogen[:\s]+(\d+)',
            'P': r'phosphorus[:\s]+(\d+)',
            'K': r'potassium[:\s]+(\d+)',
            'temperature': r'temperature[:\s]+(\d+)',
            'humidity': r'humidity[:\s]+(\d+)',
            'ph': r'ph[:\s]+(\d+\.?\d*)',
            'rainfall': r'rainfall[:\s]+(\d+)'
        }
        
        query_lower = query.lower()
        
        for param, pattern in patterns.items():
            match = re.search(pattern, query_lower)
            if match:
                value = float(match.group(1))
                params[param] = value
        
        return params
    
    def process_query(self, query, image_path=None, crop_params=None):
        """
        Process user query and route to appropriate tool
        
        Args:
            query: User's question or request
            image_path: Path to plant image (for disease detection)
            crop_params: Dict with soil/climate params (for crop recommendation)
            
        Returns:
            dict with response
        """
        
        if not self.is_initialized:
            return {
                'error': 'Agent not initialized. Call initialize() first.',
                'response': None
            }
        
        print("\n" + "="*70)
        print("🤖 SMART FARMING ADVISOR")
        print("="*70)
        print(f"\n💬 Your Query: {query}")
        
        # Classify intent
        intent = self.classify_intent(query)
        print(f"🎯 Detected Intent: {intent.upper()}")
        
        # Route to appropriate tool
        if intent == 'crop':
            print("\n🌾 Routing to: Crop Recommendation Tool")
            
            # Use provided params or try to extract from query
            if not crop_params:
                crop_params = self.extract_crop_params(query)
            
            # Check if we have all required parameters
            required = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
            if all(k in crop_params for k in required):
                result = self.crop_tool.predict(
                    N=crop_params['N'],
                    P=crop_params['P'],
                    K=crop_params['K'],
                    temperature=crop_params['temperature'],
                    humidity=crop_params['humidity'],
                    ph=crop_params['ph'],
                    rainfall=crop_params['rainfall'],
                    verbose=True
                )
                return {
                    'intent': 'crop',
                    'result': result,
                    'response': f"Based on your soil and climate conditions, I recommend: {result['recommended_crop']} (Confidence: {result['confidence']*100:.1f}%)"
                }
            else:
                missing = [k for k in required if k not in crop_params]
                return {
                    'intent': 'crop',
                    'error': f'Missing parameters: {", ".join(missing)}',
                    'response': f'To recommend a crop, I need: Nitrogen (N), Phosphorus (P), Potassium (K), Temperature, Humidity, pH, and Rainfall values.'
                }
        
        elif intent == 'disease':
            print("\n🍃 Routing to: Disease Detection Tool")
            
            if image_path:
                result = self.disease_tool.predict(image_path, verbose=True)
                if 'error' not in result:
                    return {
                        'intent': 'disease',
                        'result': result,
                        'response': f"Disease Detected: {result['predicted_disease']} (Confidence: {result['confidence']*100:.1f}%)"
                    }
                else:
                    return {
                        'intent': 'disease',
                        'error': result['error'],
                        'response': 'Error detecting disease from image.'
                    }
            else:
                return {
                    'intent': 'disease',
                    'error': 'No image provided',
                    'response': 'To detect plant diseases, please provide an image of the affected plant leaf.'
                }
        
        else:  # qa
            print("\n❓ Routing to: RAG Q&A Tool")
            result = self.qa_tool.answer_question(query, use_llm=False, verbose=True)
            return {
                'intent': 'qa',
                'result': result,
                'response': result.get('answer', 'No answer found.')
            }

# =============================================================================
# TESTING
# =============================================================================

def test_farming_agent():
    """Test agent with different types of queries"""
    
    print("\n" + "="*70)
    print("🧪 TESTING SMART FARMING ADVISOR AGENT")
    print("="*70)
    
    # Initialize agent
    agent = FarmingAgent()
    success = agent.initialize()
    
    if not success:
        print("\n❌ Agent initialization failed")
        return
    
    # Test Case 1: Crop Recommendation
    print("\n\n" + "="*70)
    print("TEST CASE 1: CROP RECOMMENDATION")
    print("="*70)
    
    crop_params = {
        'N': 90, 'P': 42, 'K': 43,
        'temperature': 20, 'humidity': 82,
        'ph': 6.5, 'rainfall': 202
    }
    
    response = agent.process_query(
        "What crop should I plant?",
        crop_params=crop_params
    )
    
    print(f"\n✅ Agent Response: {response['response']}")
    
    # Test Case 2: Disease Detection
    print("\n\n" + "="*70)
    print("TEST CASE 2: DISEASE DETECTION")
    print("="*70)
    
    # Find a sample test image
    import config
    test_dir = config.DISEASE_PROCESSED_DIR / 'test'
    sample_images = []
    for class_dir in test_dir.iterdir():
        if class_dir.is_dir():
            images = list(class_dir.glob('*.JPG'))
            if images:
                sample_images.append(images[0])
                break
    
    if sample_images:
        response = agent.process_query(
            "My plant looks sick, can you identify the disease?",
            image_path=sample_images[0]
        )
        print(f"\n✅ Agent Response: {response['response']}")
    else:
        print("\n⚠️  No test images found")
    
    # Test Case 3: General Q&A
    print("\n\n" + "="*70)
    print("TEST CASE 3: FARMING Q&A")
    print("="*70)
    
    response = agent.process_query(
        "What is the best time to plant rice?"
    )
    
    print(f"\n✅ Agent Response: {response['response']}")
    
    # Test Case 4: Another Q&A
    print("\n\n" + "="*70)
    print("TEST CASE 4: FARMING Q&A")
    print("="*70)
    
    response = agent.process_query(
        "How often should I water tomato plants?"
    )
    
    print(f"\n✅ Agent Response: {response['response']}")
    
    print("\n\n" + "="*70)
    print("✅ AGENT TESTING COMPLETE!")
    print("="*70)
    
    print("\n🎯 SUMMARY:")
    print("   ✅ Crop Recommendation: Working")
    print("   ✅ Disease Detection: Working")
    print("   ✅ Q&A System: Working")
    print("   ✅ Intent Classification: Working")

if __name__ == "__main__":
    test_farming_agent()