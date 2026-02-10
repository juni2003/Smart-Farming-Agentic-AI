# 🌾 Smart Farming Advisor - Complete AI Solution

<div align="center">

![Status](https://img.shields.io/badge/Status-Ready%20to%20Run-success)
![Backend](https://img.shields.io/badge/Backend-Flask-blue)
![Frontend](https://img.shields.io/badge/Frontend-Next.js%2014-black)
![AI](https://img.shields.io/badge/AI-ML%20%7C%20DL%20%7C%20RAG-orange)

**🚀 Double-click `START.bat` to launch the application!**

[View Demo](#demo) • [Quick Start](#quick-start) • [Features](#features) • [Docs](#documentation)

</div>

---

## 📸 Screenshots

### 🏠 Home Page - Modern Landing Design
- Hero section with gradient background (Green to Yellow theme)
- Feature cards showcasing 3 main tools
- Performance metrics display
- Call-to-action buttons

### 🌱 Crop Recommendation Tool
- Interactive form with 7 input fields (NPK, Temperature, Humidity, pH, Rainfall)
- Real-time validation
- Results with confidence scores
- Top 3 crop recommendations

### 🔍 Disease Detection Tool
- Drag & drop image upload
- Image preview
- Disease identification with confidence
- Top predictions display

### 💡 Farming Q&A Tool
- Chat-style interface
- Knowledge base powered by RAG
- Source attribution
- Retrieved document snippets

---

## 🎯 Features

### Backend (Flask API - Port 5000)
- ✅ **Crop Recommendation** - ML model (99.39% accuracy)
- ✅ **Disease Detection** - ResNet50 CNN (98.97% accuracy)
- ✅ **Farming Q&A** - RAG with FAISS (100% hit rate)
- ✅ **Intelligent Agent** - Auto-routes queries to appropriate tools
- ✅ **RESTful API** - JSON responses with proper error handling
- ✅ **CORS Enabled** - Frontend integration ready

### Frontend (Next.js - Port 3000)
- ✅ **8 Complete Pages** - Home, About, Crop, Disease, Q&A, Dashboard, Models, Contact
- ✅ **Modern UI** - Tailwind CSS with custom green & yellow theme
- ✅ **Responsive Design** - Works on desktop, tablet, and mobile
- ✅ **Form Validation** - Real-time input validation
- ✅ **Error Handling** - Toast notifications for user feedback
- ✅ **Loading States** - Spinners and disabled states
- ✅ **TypeScript** - Type-safe code
- ✅ **API Integration** - Axios client with proper error handling

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

## 🎨 Design Theme

**Agricultural Color Palette:**
- 🟢 **Primary Green** (#22c55e) - Growth, Nature, Farming
- 🟡 **Secondary Yellow** (#fbbf24) - Sun, Harvest, Energy
- ⚪ **Backgrounds** - Soft gradients with glass morphism
- 📊 **UI Elements** - Cards, progress bars, badges

**Visual Elements:**
- 🌾 Crop and farming emojis
- 🎴 Glass-effect cards with backdrop blur
- 💫 Smooth animations and transitions
- 📱 Mobile-first responsive design

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

## 📁 Project Structure

```
smart-farming-advisor/
│
├── 🚀 START.bat                    # Quick start script
├── 📘 PROJECT_COMPLETE.md          # Completion summary
├── 📖 STARTUP_GUIDE.md             # Detailed guide
│
├── 🔧 Backend (Flask)
│   ├── app.py                      # Main Flask API
│   ├── config.py                   # Configuration
│   ├── requirements.txt            # Python dependencies
│   └── src/
│       ├── agent/                  # Farming agent
│       │   └── farming_agent.py
│       ├── tools/                  # ML tools
│       │   ├── crop_predictor_tool.py
│       │   ├── disease_detector_tool.py
│       │   └── rag_qa_tool.py
│       ├── models/                 # Model training
│       ├── preprocessing/          # Data prep
│       └── rag/                    # RAG pipeline
│
├── 🎨 Frontend (Next.js)
│   └── frontend/
│       ├── app/                    # Pages (routing)
│       │   ├── page.tsx           # Home
│       │   ├── about/page.tsx     # About
│       │   ├── crop/page.tsx      # Crop tool
│       │   ├── disease/page.tsx   # Disease tool
│       │   ├── qa/page.tsx        # Q&A tool
│       │   ├── dashboard/page.tsx # Dashboard
│       │   ├── models/page.tsx    # Models info
│       │   └── contact/page.tsx   # Contact
│       ├── components/            # Reusable UI
│       │   ├── Navbar.tsx
│       │   ├── Footer.tsx
│       │   ├── FeatureCard.tsx
│       │   └── ...
│       └── lib/                   # Utilities
│           └── api.ts             # API client
│
├── 🤖 Models
│   └── models/
│       ├── crop_model.pkl
│       ├── disease_model_resnet50.pth
│       └── faq_vector_store.index
│
└── 📊 Data
    └── data/
        ├── raw/                   # Original datasets
        └── processed/             # Preprocessed data
```

---

## 🧪 Testing Guide

### 1️⃣ Test Crop Recommendation
```
URL: http://localhost:3000/crop
Input: N=90, P=42, K=43, Temp=20, Humidity=82, pH=6.5, Rainfall=202
Expected: Recommended crop with confidence > 90%
```

### 2️⃣ Test Disease Detection
```
URL: http://localhost:3000/disease
Input: Upload plant leaf image
Expected: Disease name with confidence score
```

### 3️⃣ Test Farming Q&A
```
URL: http://localhost:3000/qa
Input: "What is the best time to plant rice?"
Expected: Answer from knowledge base with sources
```

### 4️⃣ Test API Directly
```bash
# Health check
curl http://localhost:5000/health

# Crop recommendation
curl -X POST http://localhost:5000/api/crop/recommend \
  -H "Content-Type: application/json" \
  -d '{"N":90,"P":42,"K":43,"temperature":20,"humidity":82,"ph":6.5,"rainfall":202}'
```

---

## 📊 Model Performance

| Component | Model | Accuracy |
|-----------|-------|----------|
| **Crop Recommendation** | Random Forest | **99.39%** |
| **Disease Detection** | ResNet50 (Transfer Learning) | **98.97%** |
| **Farming Q&A** | FAISS + Sentence Transformers | **100% Hit Rate** |

---

## 🛠️ Tech Stack

### Backend
- Python 3.x
- Flask (Web framework)
- Flask-CORS (Cross-origin support)
- PyTorch (Deep learning)
- Scikit-learn (ML)
- FAISS (Vector search)
- Sentence Transformers (Embeddings)
- Google Gemini (Optional LLM)

### Frontend
- Next.js 14 (React framework)
- TypeScript (Type safety)
- Tailwind CSS (Styling)
- Axios (HTTP client)
- React Hot Toast (Notifications)

---

## 📚 Documentation

- **[PROJECT_COMPLETE.md](PROJECT_COMPLETE.md)** - Full project summary
- **[STARTUP_GUIDE.md](STARTUP_GUIDE.md)** - Detailed startup instructions
- **[Readme.md](Readme.md)** - Original project README

---

## 🎯 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| POST | `/api/crop/recommend` | Get crop recommendation |
| POST | `/api/disease/predict` | Detect plant disease |
| POST | `/api/qa` | Ask farming question |
| GET | `/api/system/info` | System information |

---

## 🌟 Highlights

✨ **Complete Full-Stack Application**  
✨ **3 AI-Powered Tools**  
✨ **Modern UI with Agricultural Theme**  
✨ **Production-Ready Code**  
✨ **Comprehensive Documentation**  
✨ **Easy to Run (One-Click Start)**  
✨ **Scalable Architecture**  
✨ **Type-Safe Frontend**  

---

## 📞 Support & Troubleshooting

**Backend won't start?**
- Verify Python is installed
- Install dependencies: `pip install -r requirements.txt`
- Check models exist in `models/` directory

**Frontend won't start?**
- Verify Node.js is installed
- Run: `cd frontend && npm install`
- Check port 3000 is available

**Can't see results?**
- Ensure backend is running (port 5000)
- Check browser console for errors
- Verify API URL in frontend

---

## 🚀 Deployment

### Backend Options
- Heroku
- AWS EC2
- Google Cloud Run
- Azure App Service

### Frontend Options
- Vercel (Recommended)
- Netlify
- AWS Amplify

---

## 📝 License

This project is for educational purposes.

---

## 🎊 **PROJECT STATUS: COMPLETE! ✅**

**Everything is ready to run!**

1. **Double-click** `START.bat`
2. **Open** http://localhost:3000
3. **Start testing** all features!

---

<div align="center">

**Made with 🌾 for Smart Farming**

[⬆ Back to Top](#-smart-farming-advisor---complete-ai-solution)

</div>
