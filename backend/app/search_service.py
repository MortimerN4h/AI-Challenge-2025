import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import asyncio
from pathlib import Path
import re

class SearchService:
    def __init__(self):
        print("Đang khởi tạo SearchService...")
        # --- Cấu hình đường dẫn ---
        BASE_DIR = Path(__file__).resolve().parent.parent.parent
        DB_DIR = BASE_DIR / "static_data" / "db"
        
        self.METADATA_DB_FILE = DB_DIR / "metadata.json"
        self.FAISS_INDEX_FILE = str(DB_DIR / "faiss_index.bin")
        self.FRAME_LIST_FILE = DB_DIR / "frame_list.txt"
        self.CLIP_MODEL_NAME = 'clip-ViT-B-32-multilingual-v1'

        self.metadata = {}
        self.index = None
        self.frame_list = []
        self.model = None

        self._load_resources()

    def _load_resources(self):
        try:
            print("1. Đang tải metadata...")
            self.metadata = json.loads(self.METADATA_DB_FILE.read_text(encoding='utf-8'))
            
            print("2. Đang tải frame list...")
            self.frame_list = self.FRAME_LIST_FILE.read_text().splitlines()
            
            print("3. Đang tải FAISS index...")
            self.index = faiss.read_index(self.FAISS_INDEX_FILE)

            print("4. Đang tải model CLIP...")
            self.model = SentenceTransformer(self.CLIP_MODEL_NAME)
            
            print("--- SearchService đã sẵn sàng! ---")

        except FileNotFoundError as e:
            print(f"LỖI: Không tìm thấy tệp database cần thiết: {e.filename}")
            print("Vui lòng chạy script 'scripts/build_database.py' trước khi khởi động server.")
        except Exception as e:
            print(f"Lỗi nghiêm trọng khi tải resources: {e}")

    def is_ready(self):
        """Kiểm tra xem tất cả tài nguyên đã được tải thành công chưa."""
        return all([self.metadata, self.index, self.frame_list, self.model])

    async def search(self, query: str, k: int):
        if not self.is_ready():
            return []

        # --- Bước 1: Tìm kiếm sơ bộ bằng FAISS (Candidate Retrieval) ---
        text_embedding = self.model.encode([query])
        faiss.normalize_L2(text_embedding)
        
        # Lấy top k*3 ứng viên để có không gian cho việc re-ranking
        num_candidates = k * 3
        distances, indices = self.index.search(text_embedding, num_candidates)
        
        candidates = []
        for i in range(len(indices[0])):
            idx = indices[0][i]
            if idx != -1: # FAISS có thể trả về -1 nếu không có đủ kết quả
                candidates.append({
                    "frame_id": self.frame_list[idx],
                    "clip_score": float(distances[0][i])
                })

        # --- Bước 2: Sắp xếp lại (Re-ranking) ---
        # Đây là một logic re-ranking đơn giản, bạn có thể làm nó phức tạp và thông minh hơn
        reranked_results = self._rerank(query, candidates)

        # Trả về top k kết quả cuối cùng
        return reranked_results[:k]

    def _rerank(self, query: str, candidates: list):
        query_words = set(re.split(r'\s+', query.lower()))

        for item in candidates:
            frame_id = item["frame_id"]
            frame_meta = self.metadata.get(frame_id, {})
            
            metadata_score = 0
            
            # Cộng điểm nếu từ khóa query xuất hiện trong OCR text
            ocr_text = frame_meta.get("ocr_text", "").lower()
            if any(word in ocr_text for word in query_words):
                metadata_score += 0.2
                
            # Cộng điểm nếu từ khóa query khớp với tên object
            objects = [obj.lower() for obj in frame_meta.get("objects", [])]
            if any(word in objects for word in query_words):
                metadata_score += 0.3 # Ưu tiên object hơn
                
            # Cộng điểm nếu từ khóa query khớp với bối cảnh
            places = [place.lower() for place in frame_meta.get("places", [])]
            if any(word in places for word in query_words):
                metadata_score += 0.25

            # Tính điểm cuối cùng: kết hợp điểm CLIP và điểm metadata
            # Trọng số có thể được tùy chỉnh để đạt kết quả tốt nhất
            item['score'] = (0.6 * item['clip_score']) + (0.4 * metadata_score)

        # Sắp xếp lại danh sách ứng viên dựa trên điểm số cuối cùng
        candidates.sort(key=lambda x: x['score'], reverse=True)
        return candidates

# Khởi tạo một instance duy nhất của SearchService khi ứng dụng khởi động
search_service = SearchService()