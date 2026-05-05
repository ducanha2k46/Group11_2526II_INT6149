# Group11_2526II_INT6149

Đồ án môn học Xử lý Dữ liệu Lớn (Big Data) - **Nhóm 11**. 

Dự án này xây dựng một hệ thống phân tích dữ liệu lớn (Big Data Analytics Framework) để dự đoán rủi ro tái nhập viện trong vòng 30 ngày của bệnh nhân tiểu đường. Hệ thống sử dụng cụm tính toán phân tán **Apache Spark** làm lõi xử lý học máy và **FastAPI** để phục vụ giao diện người dùng theo thời gian thực.

## 🌟 Các tính năng chính
- **Xử lý dữ liệu quy mô lớn:** Xử lý tập dữ liệu UCI Diabetes 130-US Hospitals với ~100.000 hồ sơ bệnh án và 50 trường đặc trưng.
- **Spark ML Pipeline:** Luồng tiền xử lý dữ liệu tự động gồm 37 bước (Missing Value Imputation, StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler).
- **Học máy phân tán:** Giải quyết bài toán mất cân bằng lớp (Class Imbalance) và huấn luyện mô hình **Random Forest** phân tán trên bộ nhớ (In-memory computation).
- **Hệ thống API RESTful:** Đóng gói toàn bộ mô hình Spark vào FastAPI, cung cấp tốc độ phản hồi cực thấp (Latency < 500ms).
- **Clinical Dashboard:** Giao diện người dùng trực quan, thân thiện cho nhân viên y tế, tích hợp cơ chế phân tầng rủi ro minh bạch (Hiển thị % rủi ro).

## 🏛️ Kiến trúc Hệ thống
1. **Data Source:** CSV Dataset (UCI Machine Learning Repository).
2. **Data Ingestion & Processing:** PySpark DataFrame, Lazy Evaluation DAG.
3. **Modeling:** Spark MLlib (Random Forest, Gradient-Boosted Tree).
4. **Serving:** FastAPI + Uvicorn.
5. **Frontend:** HTML/CSS/Vanilla JS.

---

## ⚙️ Hướng dẫn Cài đặt & Chạy dự án (Môi trường Windows)

Vì lõi của hệ thống là Apache Spark chạy trên giả lập Hadoop của Windows, vui lòng làm theo chính xác các bước sau để đảm bảo hệ thống không bị lỗi phân quyền (Winutils/Hadoop.dll).

### Bước 1: Nếu môi trường là window
- Môi trường: Windows 10/11.
- [Java Development Kit (JDK) 11](https://www.oracle.com/java/technologies/javase-jdk11-downloads.html) (Bắt buộc phải là Java 11).
- Python 3.9 hoặc 3.10.
- Các file Hadoop giả lập cho Windows (winutils.exe, hadoop.dll).

### Bước 2: Thiết lập Biến môi trường (Environment Variables)
Đảm bảo bạn đã khai báo đúng các đường dẫn này trong biến môi trường của hệ thống hoặc ngay đầu file `app.py`:
```python
os.environ['JAVA_HOME'] = r'C:\Program Files\Java\jdk-11.0.30' # Sửa theo máy bạn
os.environ['HADOOP_HOME'] = r'D:\Hadoop' # Thư mục chứa folder bin có winutils
os.environ['SPARK_LOCAL_IP'] = '127.0.0.1'
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable
Lưu ý Fix lỗi lưu mô hình trên Windows: Hãy copy file hadoop.dll từ thư mục Hadoop/bin và dán thẳng vào C:\Windows\System32.

### Bước 3: Cài đặt thư viện Python
Mở Terminal/CMD, di chuyển vào thư mục dự án và kích hoạt môi trường ảo (Virtual Environment), sau đó chạy:

```
pip install pyspark==3.5.1 fastapi uvicorn pydantic
```

### Bước 4: Khởi chạy Máy chủ AI
Sử dụng Uvicorn để bật server FastAPI:

```
uvicorn app:app --reload
```
Ghi chú: Ở lần chạy đầu tiên, Spark sẽ mất khoảng 10-15 giây để gọi máy ảo Java (JVM) và nạp toàn bộ 37 bước Pipeline cùng mô hình Random Forest vào RAM.

### Bước 5: Sử dụng
Mở trình duyệt và truy cập: http://127.0.0.1:8000. Giao diện Đánh giá Tái nhập viện sẽ hiện ra.

📂 Cấu trúc Thư mục (Tree)
Plaintext
Group11_2526II_INT6149/
│
├── diabetes_prediction.ipynb       # Notebook phân tích, EDA và huấn luyện mô hình Spark
├── app.py                          # Main API Server (FastAPI + PySpark loader)
├── index.html                      # Giao diện Frontend (Clinical Dashboard)
├── README.md                       # Tài liệu hướng dẫn
│
├── diabetes+130-us+hospitals/      # (Thư mục ẩn) Chứa dữ liệu CSV gốc
├── env/                            # Môi trường ảo Python
│
└── spark_models_diabetes_readmission/
    ├── preprocess_model/           # Pipeline tiền xử lý đã lưu (StringIndexer, Encoder...)
    └── best_model_Random_Forest/   # Mô hình Random Forest đã huấn luyện