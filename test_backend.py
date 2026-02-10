"""
Test the backend components without Flask
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

print("\n" + "="*70)
print("🧪 TESTING BACKEND COMPONENTS")
print("="*70)

# Test 1: Configuration
print("\n1️⃣  Testing Configuration...")
try:
    import config
    print(f"   ✅ Config loaded")
    print(f"   - Data dir: {config.DATA_DIR}")
    print(f"   - Models dir: {config.MODELS_DIR}")
    print(f"   - Google API Key: {'✓ Set' if config.GOOGLE_API_KEY else '✗ Not set'}")
except Exception as e:
    print(f"   ❌ Error: {e}")

# Test 2: Crop Predictor Tool
print("\n2️⃣  Testing Crop Predictor Tool...")
try:
    from src.tools.crop_predictor_tool import CropPredictorTool
    tool = CropPredictorTool()
    success = tool.load_model()
    if success:
        print(f"   ✅ Crop model loaded successfully")
        
        # Test prediction
        result = tool.predict(
            N=90, P=42, K=43,
            temperature=20, humidity=82,
            ph=6.5, rainfall=202,
            verbose=False
        )
        print(f"   ✅ Prediction works: {result['recommended_crop']}")
    else:
        print(f"   ❌ Failed to load crop model")
except Exception as e:
    print(f"   ❌ Error: {e}")
    import traceback
    traceback.print_exc()

# Test 3: Disease Detector Tool
print("\n3️⃣  Testing Disease Detector Tool...")
try:
    from src.tools.disease_detector_tool import DiseaseDetectorTool
    tool = DiseaseDetectorTool()
    success = tool.load_model()
    if success:
        print(f"   ✅ Disease model loaded successfully")
        print(f"   - Classes: {len(tool.class_names)}")
    else:
        print(f"   ❌ Failed to load disease model")
except Exception as e:
    print(f"   ❌ Error: {e}")
    import traceback
    traceback.print_exc()

# Test 4: RAG Tool
print("\n4️⃣  Testing RAG Q&A Tool...")
try:
    from src.tools.rag_qa_tool import RAGQATool
    tool = RAGQATool()
    success = tool.initialize()
    if success:
        print(f"   ✅ RAG tool initialized successfully")
        
        # Test Q&A
        result = tool.answer_question("What is the best time to plant rice?", verbose=False)
        if 'error' not in result:
            print(f"   ✅ Q&A works: Got answer")
        else:
            print(f"   ⚠️  Q&A error: {result['error']}")
    else:
        print(f"   ⚠️  RAG tool initialization had issues")
except Exception as e:
    print(f"   ❌ Error: {e}")
    import traceback
    traceback.print_exc()

# Test 5: Farming Agent
print("\n5️⃣  Testing Farming Agent...")
try:
    from src.agent.farming_agent import FarmingAgent
    agent = FarmingAgent()
    success = agent.initialize()
    if success:
        print(f"   ✅ Agent initialized successfully")
        print(f"   - Crop tool: {'✓' if agent.crop_tool.is_loaded else '✗'}")
        print(f"   - Disease tool: {'✓' if agent.disease_tool.is_loaded else '✗'}")
        print(f"   - QA tool: {'✓' if agent.qa_tool.is_initialized else '✗'}")
    else:
        print(f"   ❌ Failed to initialize agent")
except Exception as e:
    print(f"   ❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)
print("✅ BACKEND TEST COMPLETE")
print("="*70 + "\n")
