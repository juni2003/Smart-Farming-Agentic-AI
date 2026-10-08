# 🌾 Smart Farming Advisor - Complete Setup & Run Guide

## ✅ SYSTEM STATUS

Your Smart Farming Advisor system is **READY TO RUN**! Here's what we've built:

### Backend (Flask API) ✓
- **Location**: `c:\Users\LAPTOP CLINIC\Documents\Projects\smart-farming-advisor\app.py`
- **Framework**: Flask + Flask-CORS
- **Port**: 5000
- **Endpoints**:
  - `GET /health` - Health check
  - `POST /api/crop/recommend` - Crop recommendation
  - `POST /api/disease/predict` - Disease detection (upload image)
  - `POST /api/qa` - Farming Q&A
  - `GET /api/system/info` - System information

### Frontend (Next.js + Tailwind CSS) ✓
- **Location**: `c:\Users\LAPTOP CLINIC\Documents\Projects\smart-farming-advisor\frontend\`
- **Framework**: Next.js 14 + TypeScript
- **Styling**: Tailwind CSS (Green & Yellow theme)
- **Port**: 3000
- **Pages**:
  1. **Home** (`/`) - Landing page with features
  2. **About** (`/about`) - Project details
  3. **Crop Advisor** (`/crop`) - Crop recommendation tool
  4. **Disease Detection** (`/disease`) - Plant disease identifier
  5. **Farming Q&A** (`/qa`) - Ask farming questions
  6. **Dashboard** (`/dashboard`) - Results dashboard
  7. **Models Info** (`/models`) - Model metrics
  8. **Contact** (`/contact`) - Feedback form

---

## 🚀 HOW TO RUN

### STEP 1: Start Backend Server

Open a terminal and run:

```bash
cd "c:\Users\LAPTOP CLINIC\Documents\Projects\smart-farming-advisor"
python app.py
```

You should see:
```
🚀 STARTING SMART FARMING ADVISOR API
✅ API server starting on http://localhost:5000
```

**Keep this terminal running!**

---

### STEP 2: Start Frontend

Open a **NEW** terminal and run:

```bash
cd "c:\Users\LAPTOP CLINIC\Documents\Projects\smart-farming-advisor\frontend"
npm install
npm run dev
```

You should see:
```
- Local:        http://localhost:3000
```

---

### STEP 3: Access the Application

Open your browser and go to:
**http://localhost:3000**

---

## 🎨 DESIGN THEME

**Colors:**
- Primary Green: `#22c55e` (for nature/farming)
- Secondary Yellow: `#fbbf24` (for sun/crops)
- Background: Gradient from green to yellow
- Glass effect cards with backdrop blur

**UI Features:**
- Modern, clean design
- Responsive (works on mobile)
- Smooth transitions
- Icons and emojis for visual appeal

---

## 🧪 TESTING THE SYSTEM

### Test 1: Crop Recommendation
1. Go to http://localhost:3000/crop
2. Enter values:
   - N: 90
   - P: 42
   - K: 43
   - Temperature: 20
   - Humidity: 82
   - pH: 6.5
   - Rainfall: 202
3. Click "Get Recommendation"
4. Should see recommended crop with confidence score

### Test 2: Disease Detection
1. Go to http://localhost:3000/disease
2. Upload a plant leaf image
3. Click "Detect Disease"
4. Should see disease prediction with confidence

### Test 3: Farming Q&A
1. Go to http://localhost:3000/qa
2. Ask: "What is the best time to plant rice?"
3. Should get relevant answer from knowledge base

---

## 📁 PROJECT STRUCTURE

```
smart-farming-advisor/
├── app.py                 # Flask API backend
├── config.py              # Configuration
├── requirements.txt       # Python dependencies
├── data/                  # Datasets
│   ├── processed/         # Preprocessed data
│   └── raw/               # Original data
├── models/                # Trained ML models
│   ├── crop_model.pkl
│   ├── disease_model_resnet50.pth
│   └── faq_vector_store.index
├── src/                   # Source code
│   ├── agent/             # Farming agent
│   ├── tools/             # Crop, Disease, RAG tools
│   ├── models/            # Model training scripts
│   ├── preprocessing/     # Data preprocessing
│   └── rag/               # RAG pipeline
└── frontend/              # Next.js frontend
    ├── app/               # Pages
    │   ├── page.tsx       # Home
    │   ├── about/
    │   ├── crop/
    │   ├── disease/
    │   ├── qa/
    │   ├── dashboard/
    │   ├── models/
    │   └── contact/
    ├── components/        # Reusable components
    └── lib/               # Utilities & API client
```

---

## 🔧 TROUBLESHOOTING

### Backend won't start?
1. Check Python is installed: `python --version`
2. Install dependencies: `pip install -r requirements.txt`
3. Check if models exist in `models/` directory

### Frontend won't start?
1. Check Node.js is installed: `node --version`
2. Install dependencies: `cd frontend && npm install`
3. Check if port 3000 is available

### Can't connect frontend to backend?
1. Make sure backend is running on port 5000
2. Check `.env` file has `NEXT_PUBLIC_API_URL=http://localhost:5000`
3. Look for CORS errors in browser console

---

## 📊 MODELS PERFORMANCE

| Model | Type | Accuracy |
|-------|------|----------|
| Crop Recommendation | Random Forest | 99.39% |
| Disease Detection | ResNet50 | 98.97% |
| RAG Q&A | FAISS + Transformers | 100% Hit Rate |

---

## 🎯 NEXT STEPS

1. **Test all features** from the browser
2. **Customize the UI** colors/images as needed
3. **Add your own data** to improve models
4. **Deploy** to cloud (Vercel for frontend, Python hosting for backend)

---

## 📝 QUICK COMMANDS

```bash
# Backend
cd "c:\Users\LAPTOP CLINIC\Documents\Projects\smart-farming-advisor"
python app.py

# Frontend (new terminal)
cd "c:\Users\LAPTOP CLINIC\Documents\Projects\smart-farming-advisor\frontend"
npm run dev

# Access
# Frontend: http://localhost:3000
# Backend API: http://localhost:5000
```

---

## ✨ FEATURES INCLUDED

✅ **Crop Recommendation** - ML-powered crop suggestions  
✅ **Disease Detection** - Computer vision for plant diseases  
✅ **Farming Q&A** - RAG-based knowledge system  
✅ **Modern UI** - Green/Yellow theme with gradients  
✅ **Responsive Design** - Works on all devices  
✅ **API Integration** - Frontend connected to backend  
✅ **Error Handling** - Toast notifications for users  
✅ **Loading States** - User feedback during processing  

---
**SYSTEM IS READY! 🚀**

Just run the two commands above and access http://localhost:3000 in your browser!
