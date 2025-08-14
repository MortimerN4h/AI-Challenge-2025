import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from PIL import Image
from pathlib import Path
from tqdm import tqdm

# --- CẤU HÌNH ---
# Các đường dẫn tương đối so với thư mục gốc của project
KEYFRAME_DIR = Path("static_data/keyframes")
OUTPUT_DIR = Path("static_data/model_outputs")
DB_DIR = Path("static_data/db")

# Các tệp output từ Colab (bạn cần đảm bảo đã tạo ra các tệp này)
OCR_FILE = OUTPUT_DIR / "ocr.json"
OBJECTS_FILE = OUTPUT_DIR / "objects.json"
PLACES_FILE = OUTPUT_DIR / "places.json"

# Các tệp database sẽ được tạo ra
METADATA_DB_FILE = DB_DIR / "metadata.json"
FAISS_INDEX_FILE = DB_DIR / "faiss_index.bin"

# Model CLIP để tạo image embedding (nếu bạn chưa tạo từ Colab)
CLIP_MODEL_NAME = 'clip-ViT-B-32-multilingual-v1'

def build_database():
    """
    Hàm chính để tổng hợp output từ các model và xây dựng database.
    """
    print(" Bắt đầu quá trình xây dựng database ".center(80, "="))

    # Tạo các thư mục cần thiết nếu chưa có
    DB_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)


    # --- Bước 1: Tổng hợp Metadata từ các file JSON ---
    print("\n [1/3] Đang tổng hợp metadata...")
    
    # Lấy danh sách tất cả các frame ảnh
    frame_files = sorted([f.name for f in KEYFRAME_DIR.glob("*.jpg")])
    if not frame_files:
        print(f"Lỗi: Không tìm thấy ảnh nào trong thư mục '{KEYFRAME_DIR}'. Vui lòng thêm ảnh vào.")
        return

    print(f"Tìm thấy {len(frame_files)} keyframes.")

    # Tải dữ liệu từ các file JSON output
    ocr_data = json.loads(OCR_FILE.read_text()) if OCR_FILE.exists() else {}
    objects_data = json.loads(OBJECTS_FILE.read_text()) if OBJECTS_FILE.exists() else {}
    places_data = json.loads(PLACES_FILE.read_text()) if PLACES_FILE.exists() else {}
    
    # Tạo metadata tổng hợp
    metadata = {}
    for frame_id in tqdm(frame_files, desc="Tổng hợp metadata"):
        metadata[frame_id] = {
            "ocr_text": ocr_data.get(frame_id, ""),
            "objects": objects_data.get(frame_id, []),
            "places": places_data.get(frame_id, []),
        }
    
    # Lưu file metadata.json
    METADATA_DB_FILE.write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
    print(f"Đã lưu metadata tổng hợp vào '{METADATA_DB_FILE}'")


    # --- Bước 2: Tạo Image Embeddings và xây dựng FAISS Index ---
    print(f"\n [2/3] Đang tạo image embeddings và xây dựng FAISS index...")
    
    print(f"Tải model CLIP: '{CLIP_MODEL_NAME}'...")
    clip_model = SentenceTransformer(CLIP_MODEL_NAME)
    
    image_paths = [KEYFRAME_DIR / f for f in frame_files]
    pil_images = [Image.open(path) for path in image_paths]

    print("Bắt đầu mã hóa ảnh (đây là bước tốn thời gian nhất)...")
    image_embeddings = clip_model.encode(
        pil_images, 
        batch_size=32, 
        convert_to_numpy=True, 
        show_progress_bar=True
    )
    
    # Chuẩn hóa L2 cho tìm kiếm cosine similarity
    faiss.normalize_L2(image_embeddings)
    
    d = image_embeddings.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(image_embeddings)
    
    print(f"Xây dựng FAISS index thành công với {index.ntotal} vectors.")
    
    # Lưu FAISS index
    faiss.write_index(index, str(FAISS_INDEX_FILE))
    print(f"Đã lưu FAISS index vào '{FAISS_INDEX_FILE}'")


    # --- Bước 3: Lưu danh sách frame_id để map với index ---
    # FAISS chỉ lưu index dạng số (0, 1, 2,..), ta cần file này để biết index 0 tương ứng với ảnh nào.
    print(f"\n [3/3] Đang lưu danh sách frame_id...")
    frame_list_path = DB_DIR / "frame_list.txt"
    frame_list_path.write_text("\n".join(frame_files))
    print(f"Đã lưu danh sách frame ID vào '{frame_list_path}'")


    print("\n" + " Hoàn tất! Database đã sẵn sàng cho backend. ".center(80, "="))


if __name__ == "__main__":
    # Hướng dẫn cho người dùng
    print("Script này sẽ xây dựng database từ output của các model AI.")
    print("Vui lòng đảm bảo bạn đã:")
    print(f"1. Đặt các ảnh keyframe vào thư mục: '{KEYFRAME_DIR}'")
    print(f"2. Chạy các model trên Colab và lưu kết quả (ocr.json, objects.json, places.json) vào thư mục: '{OUTPUT_DIR}'")
    input("\nNhấn Enter để bắt đầu...")
    build_database()