from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from pathlib import Path

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

# --- Phục vụ File Tĩnh (Static Files) ---
# Phục vụ các ảnh keyframes tại /keyframes
app.mount("/keyframes", StaticFiles(directory=KEYFRAME_DIR), name="keyframes")

# Phục vụ các file của frontend (HTML, CSS, JS) tại đường dẫn gốc
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")