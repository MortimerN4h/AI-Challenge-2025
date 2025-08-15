from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from pathlib import Path
import asyncio
from googletrans import Translator

# Import dịch vụ tìm kiếm
from .search_service import search_service

# --- Khởi tạo ứng dụng FastAPI ---
app = FastAPI(title="Video Retrieval API")

# --- Cấu hình đường dẫn ---
BASE_DIR = Path(__file__).resolve().parent.parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"
KEYFRAME_DIR = BASE_DIR / "static_data" / "keyframes"

# --- Model Dữ liệu (Data Model) ---
class SearchQuery(BaseModel):
    query: str
    k: int = 24

# --- API Endpoint ---
@app.post("/api/search")
async def api_search(request: SearchQuery):
    """
    API endpoint để xử lý yêu cầu tìm kiếm.
    """
    if not search_service.is_ready():
        raise HTTPException(status_code=503, detail="Dịch vụ tìm kiếm chưa sẵn sàng. Vui lòng kiểm tra logs của server.")
    
    try:
        results = await search_service.search(request.query, request.k)
        return {"query": request.query, "results": results}
    except Exception as e:
        # Ghi log lỗi ở đây nếu cần
        print(f"Lỗi khi xử lý truy vấn: {e}")
        raise HTTPException(status_code=500, detail="Đã có lỗi xảy ra trong quá trình xử lý.")

# API endpoint để dịch văn bản sử dụng LibreTranslate
@app.post("/api/translate")
async def translate(request: Request):
    data = await request.json()
    text = data.get("text", "")
    if not text:
        return JSONResponse({"error": "No text provided"}, status_code=400)
    try:
        translator = Translator()
        result = await translator.translate(text, src="vi", dest="en")
        return {"translated": result.text}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

# --- Phục vụ File Tĩnh (Static Files) ---
# Phục vụ các ảnh keyframes tại /keyframes
app.mount("/keyframes", StaticFiles(directory=KEYFRAME_DIR), name="keyframes")

# Phục vụ các file của frontend (HTML, CSS, JS) tại đường dẫn gốc
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
