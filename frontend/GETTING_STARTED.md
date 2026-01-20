# 🔮 MemoryLens - Getting Started Guide

## What You Have

A beautiful React.js frontend for your semantic photo & video search system with:

✨ **Modern Design**: Midnight Aurora theme with purple/indigo gradients
🎬 **Smooth Animations**: Framer Motion powered transitions
🔍 **Semantic Search**: Natural language queries
⚙️ **System Controls**: Rescan and health monitoring
📱 **Responsive**: Works on all devices

## Quick Start

### Option 1: Development Mode (Recommended for testing)

1. **Start the backend server** (in another terminal):
   ```bash
   # From project root
   cd /home/rogue/Desktop/solve-ease/photo-doc-data-embeddings
   # Start your backend however you normally do
   # e.g., python server/server.py or uvicorn server.server:app --reload --port 8000
   ```

2. **Start the frontend**:
   ```bash
   cd frontend
   npm run dev
   ```

3. **Open your browser**:
   Visit http://localhost:3000

### Option 2: Using the Start Script

```bash
cd frontend
./start.sh
```

### Option 3: Docker (Production Ready)

Update your `docker-compose.yml` to include the frontend:

```yaml
version: '3.8'

services:
  # Your existing services...
  
  frontend:
    build: ./frontend
    ports:
      - "3000:80"
    depends_on:
      - backend
    networks:
      - app-network

  backend:
    # Your backend service
    ports:
      - "8000:8000"
    networks:
      - app-network

networks:
  app-network:
    driver: bridge
```

Then run:
```bash
docker-compose up --build
```

## Using the Application

### 1. Initial Setup

When you first open the app:
1. Check if the server status shows "✅ Server Online"
2. If offline, start your backend server
3. Click "🔄 Rescan Media" to index your photos/videos

### 2. Searching

Simply type what you're looking for in natural language:

**Example queries:**
- "sunset over mountains"
- "person walking a dog"
- "car on highway"
- "kids playing in park"
- "food on table"
- "beach scene with palm trees"

The AI will understand semantic meaning, not just keywords!

### 3. Viewing Results

- Results show as cards with:
  - Preview image
  - Similarity score (percentage)
  - File name and path
- Hover over cards for zoom effect
- Click to view (if browser supports file:// protocol)

### 4. System Controls

**Rescan Media**: 
- Scans your media folders for new/changed files
- Updates the vector database
- Shows progress with loading indicator

**Server Status**:
- Green ✅ = Backend is online
- Red ❌ = Backend is offline
- Click to manually check connection

## Architecture Overview

```
┌─────────────┐         HTTP          ┌─────────────┐
│   Frontend  │ ◄──────────────────► │   Backend   │
│  (React)    │   /search, /controls  │  (FastAPI)  │
│  Port 3000  │                       │  Port 8000  │
└─────────────┘                       └─────────────┘
                                             │
                                             ▼
                                      ┌─────────────┐
                                      │   Qdrant    │
                                      │  VectorDB   │
                                      └─────────────┘
```

## Customization

### Change Colors

Edit `/frontend/src/App.css`:

```css
:root {
  --bg-primary: #0a0e1a;        /* Main background */
  --accent-primary: #6366f1;     /* Primary accent (purple) */
  --accent-secondary: #8b5cf6;   /* Secondary accent (indigo) */
}
```

### Change Logo/Title

Edit `/frontend/src/App.jsx`:

```jsx
<span className="app-logo-icon">🔮</span>  {/* Change emoji */}
<h1 className="app-title">MemoryLens</h1>  {/* Change title */}
```

### Modify Animations

Search speeds are in `App.jsx`:

```jsx
<motion.div
  initial={{ opacity: 0, y: 20 }}
  animate={{ opacity: 1, y: 0 }}
  transition={{ duration: 0.5, delay: 0.3 }}  // ← Adjust these
>
```

## Troubleshooting

### "Server Offline" Error

**Solution**: Make sure your FastAPI backend is running on port 8000
```bash
# Check if backend is running
curl http://localhost:8000/

# If not, start it
cd /path/to/server
python server.py
# or
uvicorn server.server:app --reload --port 8000
```

### Images Not Loading

**Issue**: Browser security blocks `file://` protocol

**Solutions**:
1. Serve images through backend API (recommended)
2. Use a local web server for images
3. Convert images to base64 in response

### CORS Errors

Add to your FastAPI backend:

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### Build Errors

```bash
cd frontend
rm -rf node_modules package-lock.json
npm install
```

### Port Already in Use

```bash
# Kill process on port 3000
lsof -ti:3000 | xargs kill -9

# Or use a different port in vite.config.js
server: {
  port: 3001  // Change this
}
```

## Development Tips

### Hot Module Replacement (HMR)

Vite provides instant updates:
- Edit any `.jsx` or `.css` file
- Save → Changes appear immediately
- No page refresh needed (usually)

### Console Debugging

Open browser DevTools (F12) to see:
- API request/response logs
- React component states
- Network traffic

### API Testing

Test backend directly:
```bash
# Test search
curl "http://localhost:8000/search?query=sunset"

# Test health
curl http://localhost:8000/

# Test rescan
curl http://localhost:8000/controls/rescan
```

## Production Deployment

### 1. Build the App

```bash
cd frontend
npm run build
```

Creates optimized files in `dist/` folder.

### 2. Preview Production Build

```bash
npm run preview
```

### 3. Deploy Options

**Static Hosting**:
- Vercel, Netlify, GitHub Pages
- Upload `dist/` folder
- Configure API proxy/CORS

**Docker**:
- Use provided Dockerfile
- Multi-stage build (Node → Nginx)
- Includes nginx config for API proxy

**Self-Hosted**:
- Any web server (Nginx, Apache, Caddy)
- Serve `dist/` folder
- Configure reverse proxy for API

## Next Steps

### Enhance Backend Integration

1. Add image serving endpoint to backend
2. Implement pagination for results
3. Add filters (date, type, etc.)

### UI Improvements

1. Add lightbox for full-size viewing
2. Implement search history
3. Add keyboard shortcuts
4. Create settings panel

### Advanced Features

1. Drag & drop image search
2. Batch operations (delete, move)
3. Collections/favorites
4. Export search results

## Support

For issues:
1. Check browser console for errors
2. Verify backend is running
3. Test API endpoints directly
4. Check this guide's troubleshooting section

Enjoy your semantic search experience! 🔮✨
