# Di chuyển vào thư mục scripts sau đó thực hiện lệnh:
pip install -r requirements.txt
python build_database.py

# Di chuyển vào backend
pip install -r requirements.txt
# Khởi động server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
# Mở web
http://localhost:8000