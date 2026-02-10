# 🔧 CONNECTION FIX - Frontend to Backend

## ✅ PROBLEM IDENTIFIED

Your frontend was trying to connect to **port 8000** but the backend is running on **port 5000**.

Error was:
```
POST http://localhost:8000/api/crop/recommend net::ERR_CONNECTION_REFUSED
```

## ✅ SOLUTION APPLIED

### Fixed Files:
1. **`frontend/lib/api.ts`** - Changed API_BASE from `localhost:8000` to `localhost:5000`
2. **`frontend/.env.local`** - Created environment file with correct API URL

---

## 🚀 NEXT STEPS - What You Need To Do

### Step 1: Clear Frontend Cache
```bash
cd frontend

# Remove node_modules and restart
rm -r node_modules
npm install
```

Or on Windows:
```bash
cd frontend
rmdir /s /q node_modules
npm install
```

### Step 2: Restart Frontend
```bash
npm run dev
```

You should see:
```
- Local:        http://localhost:3000
```

### Step 3: Test the Connection
1. Open browser: http://localhost:3000
2. Go to **Crop Tool** page (`/crop`)
3. Click **"Get Recommendation →"**
4. You should see results instead of "failed to fetch"

---

## ✅ VERIFICATION

### Backend Console Should Show:
When you submit from frontend, you should see in backend terminal:
```
127.0.0.1 - - [DATE TIME] "POST /api/crop/recommend HTTP/1.1" 200
```

(Notice: `200` = success, `405` = method error, `500` = server error)

### Frontend Network Tab Should Show:
In browser DevTools → Network tab:
```
POST http://localhost:5000/api/crop/recommend   200 OK
```

---

## 🧪 TEST THE BACKEND DIRECTLY

To verify backend is responding correctly:

```bash
# Run this test script:
python test_api.py
```

You should see:
```
✅ Health check passed
✅ Crop recommendation successful
✅ Q&A successful
✅ System info retrieved
```

---

## 📝 FILES CHANGED

```
frontend/lib/api.ts
- OLD: const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";
+ NEW: const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:5000";

frontend/.env.local (NEW FILE)
+ NEXT_PUBLIC_API_BASE=http://localhost:5000
```

---

## 🎯 HOW TO PREVENT THIS IN FUTURE

The issue was hardcoded port number. Now it uses environment variables:
- `NEXT_PUBLIC_API_BASE` can be set in `.env.local`
- Default fallback is `localhost:5000`

To change API URL for deployment, just edit:
```bash
frontend/.env.local
```

And set:
```
NEXT_PUBLIC_API_BASE=https://your-api-domain.com
```

---

## ⚡ QUICK START (After Fixes)

```bash
# Terminal 1 - Backend (should already be running)
cd smart-farming-advisor
python app.py

# Terminal 2 - Frontend (fresh start)
cd smart-farming-advisor\frontend
npm install
npm run dev

# Then open: http://localhost:3000
```

---

## ✅ EXPECTED BEHAVIOR AFTER FIX

1. **Go to Crop Tool** (`http://localhost:3000/crop`)
2. **Fill form** with example values
3. **Click button** → Should NOT show "failed to fetch"
4. **Should see** → Recommended crop with confidence score

Same for Disease Detection and Q&A tools!

---

## 🆘 IF STILL NOT WORKING

1. **Check Backend is Running:**
   ```bash
   curl http://localhost:5000/health
   ```
   Should return: `{"status": "ok", ...}`

2. **Check Frontend Port:**
   ```bash
   npm run dev
   ```
   Should show: `Local: http://localhost:3000`

3. **Check Browser Console:**
   - Press F12
   - Go to Console tab
   - Look for any error messages

4. **Check Network Tab:**
   - Press F12
   - Go to Network tab
   - Make a request
   - Click the request and check Response/Status

---

**Everything is now configured correctly! 🎉**

Just restart your frontend and it should work! ✅
