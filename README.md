# 🌾 Smart Farming Advisor (Full‑Stack) — Agentic AI for Precision Agriculture

An end‑to‑end, agentic AI system that combines **crop recommendation**, **plant disease detection**, and **farming Q&A** — now with a modern **Next.js + Tailwind** frontend and a **Flask** API backend.
<div align="center">

![Status](https://img.shields.io/badge/Status-Ready%20to%20Run-success)
![Backend](https://img.shields.io/badge/Backend-Flask-blue)
![Frontend](https://img.shields.io/badge/Frontend-Next.js%2014-black)
![AI](https://img.shields.io/badge/AI-ML%20%7C%20DL%20%7C%20RAG-orange)

**🚀 Double-click `START.bat` to launch the application!**

[View Demo](#demo)

</div>


## ✨ Highlights

- 🌾 **Crop Recommendation**: Soil + climate‑aware crop prediction
- 🍃 **Disease Detection**: Image‑based plant disease classification
- ❓ **Farming Q&A**: RAG pipeline with semantic search + optional LLM
- 🧠 **Agentic Routing**: Smart intent routing between tools
- ⚡ **Full‑Stack UI**: Next.js 14 frontend with clean UX

---

## 📊 Performance Snapshot

| Component | Model/Method | Score | Status |
|---|---|---:|---|
| Crop Recommendation | Random Forest + Feature Engineering | **99.39%** | ✅ Excellent |
| Disease Detection | ResNet50 (Transfer Learning) | **98.97%** | ✅ Research‑grade |
| RAG Q&A | FAISS + Sentence Transformers | **Hit Rate: 100%**, **MRR: 1.0** | ✅ Production‑ready |
| Agent Routing | Intent Classification | **100%** | ✅ Perfect |

---


## 📊 System Architecture

```
┌─────────────┐
│   Browser   │  ← User Interface
└──────┬──────┘
       │ HTTP
       ▼
┌──────────────────┐
│  Next.js (3000)  │  ← Frontend
│  - Pages         │
│  - Components    │
│  - API Client    │
└──────┬───────────┘
       │ REST API
       ▼
┌──────────────────┐
│  Flask (5000)    │  ← Backend
│  - Agent Router  │
│  - API Endpoints │
└──────┬───────────┘
       │
       ├──────────┬─────────────┬──────────┐
       ▼          ▼             ▼          │
  ┌────────┐ ┌─────────┐  ┌─────────┐    │
  │  Crop  │ │ Disease │  │   RAG   │    │
  │  Tool  │ │  Tool   │  │   Tool  │    │
  └────┬───┘ └────┬────┘  └────┬────┘    │
       │          │             │         │
       ▼          ▼             ▼         ▼
  ┌────────────────────────────────────────┐
  │          Trained ML Models             │
  │  - crop_model.pkl                      │
  │  - disease_model_resnet50.pth          │
  │  - faq_vector_store.index              │
  └────────────────────────────────────────┘
```


---

## 🧰 Tech Stack

**Frontend**
- Next.js 14 + TypeScript
- Tailwind CSS
- Axios / Fetch

**Backend**
- Python 3.8+
- Flask + Flask‑CORS
- PyTorch, Scikit‑learn
- FAISS + Sentence Transformers

---


## 🚀 Quick Start

### Option 1: One-Click Start (Recommended)
```bash
Double-click: START.bat
```
This opens two terminals and starts both servers automatically!

### Option 2: Manual Start

**Terminal 1 - Backend:**
```bash
cd "c:\Users\LAPTOP CLINIC\Documents\Projects\smart-farming-advisor"
python app.py
```

**Terminal 2 - Frontend:**
```bash
cd "c:\Users\LAPTOP CLINIC\Documents\Projects\smart-farming-advisor\frontend"
npm install  # First time only
npm run dev
```

**Access:** http://localhost:3000

---

## 🔗 API Endpoints

- `GET /health` - Health check
- `POST /api/crop/recommend` - Get crop recommendations
- `POST /api/disease/predict` - Detect plant disease (multipart form‑data with `file`)
- `POST /api/qa` - Ask farming questions
- `GET /api/system/info` - Get system and model information

---

## ⚙️ Environment Variables

**Backend** (optional):
```bash
GOOGLE_API_KEY=your_gemini_api_key  # For Gemini LLM (optional)
```

**Frontend**:
```bash
NEXT_PUBLIC_API_BASE=http://localhost:5000
```

---

## 📁 Project Structure

```
smart-farming-advisor/
├── app.py                       # Flask API
├── config.py                    # Configuration
├── requirements.txt             # Python dependencies
├── frontend/                    # Next.js app
│   ├── app/                     # Pages (home, crop, disease, qa, etc.)
│   ├── components/              # Reusable UI components
│   ├── lib/                     # API client & utilities
│   └── package.json             # Node dependencies
├── src/
│   ├── agent/                   # Agent router
│   ├── tools/                   # Crop / Disease / RAG tools
│   ├── models/                  # Training scripts
│   ├── preprocessing/           # Data preprocessing
│   └── evaluation/              # Evaluation scripts
├── models/                      # Trained models (large files ignored)
├── data/                        # Datasets (ignored)
├── outputs/                     # Results, plots (ignored)
└── notebooks/                   # Jupyter notebooks
```

---

## 🧪 Testing & Evaluation

**Test the backend:**
```bash
# Test agent
python src/agent/farming_agent.py

# Test individual tools
python src/tools/crop_predictor_tool.py
python src/tools/disease_detector_tool.py
python src/tools/rag_qa_tool.py

# Evaluate RAG system
python src/evaluation/rag_evaluation.py
```

Results are saved under `outputs/`.

---

## 🧠 Model Details

### 1) Crop Recommendation
- **Algorithm**: Random Forest Classifier
- **Features**: N, P, K, Temperature, Humidity, pH, Rainfall + engineered features
- **Test Accuracy**: **99.39%**
- **Training Time**: ~2 seconds

### 2) Disease Detection
- **Architecture**: ResNet50 (Transfer Learning)
- **Pretrained on**: ImageNet
- **Input**: 224×224 RGB
- **Test Accuracy**: **98.97%**
- **Training Time**: ~33 minutes (GPU)

### 3) RAG Q&A
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2`
- **Vector Store**: FAISS
- **Optional LLM**: Google Gemini 2.0 Flash
- **Performance**: 100% hit rate, MRR 1.0

---

## 📚 Datasets

| Dataset | Source | Purpose |
|---|---|---|
| Crop Recommendation | Kaggle | Soil‑based crop prediction (2,200 samples, 22 crops) |
| Plant Disease (PlantVillage) | Public | Disease classification (20,639 images, 15 diseases) |
| FAQ Knowledge Base | Custom | Farming Q&A (10 documents) |

Large datasets and model binaries are **excluded from GitHub**. See `.gitignore`.

---

## 🛡️ GitHub & Large Files

This repo intentionally ignores:
- `data/raw/` and `data/processed/` - Raw and processed datasets
- Large model weights (`*.pth`, `*.pt`, `*.onnx`) - Trained model files
- FAISS index (`*.index`) - Vector store index
- Frontend build artifacts (`frontend/.next`, `frontend/node_modules`)
- Outputs and uploads (`outputs/`, `uploads/`)

**Remember to add all these files by yourself**, If you need any help running this project contact me at email.

---

## 🎯 Key Features

✅ **Multi-modal AI System** - Text, images, and structured data  
✅ **Agentic Routing** - Intelligent query classification  
✅ **Transfer Learning** - ResNet50 pretrained on ImageNet  
✅ **RAG Implementation** - Semantic search with FAISS  
✅ **Production-Ready** - 98-99% accuracy across all models  
✅ **Modular Architecture** - Easy to extend and maintain  
✅ **Full-Stack** - Modern frontend with Next.js + Tailwind  
✅ **Comprehensive Testing** - All components tested  

---

## 🤝 Contributing

Pull requests are welcome! Please open an issue for major changes.

---

## 📄 License

MIT License

---

## 🙌 Acknowledgments

- **Datasets:**
  - [Crop Recommendation Dataset](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset) (Kaggle)
  - [PlantVillage Disease Dataset](https://www.kaggle.com/datasets/emmarex/plantdisease)
  - Farmer Support FAQ Dataset
  
- **Pretrained Models:**
  - ResNet50 (ImageNet, PyTorch)
  - sentence-transformers/all-MiniLM-L6-v2 (HuggingFace)
  
- **Tools:**
  - Google Colab (GPU training)
  - FAISS (Facebook AI Similarity Search)
  - Google Gemini API (LLM)

---

## 📬 Contact

**Author:** juni2003  
**Email:** juni.xatti@gmail.com  
**GitHub:** [@juni2003](https://github.com/juni2003)

---

⭐ **If you find this project helpful, please consider giving it a star!**
