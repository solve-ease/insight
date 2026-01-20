#!/bin/bash

# MemoryLens Frontend Launcher
echo "🔮 Starting MemoryLens Frontend..."
echo ""
echo "📦 Installing dependencies (if needed)..."
npm install

echo ""
echo "🚀 Starting development server..."
echo "Frontend will be available at: http://localhost:3000"
echo "Make sure the backend is running on: http://localhost:8000"
echo ""

npm run dev
