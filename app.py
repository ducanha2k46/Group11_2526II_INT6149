import os
import sys
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.ml.classification import RandomForestClassificationModel
from pyspark.sql import Row

# 1. Cấu hình môi trường (Giữ nguyên như lúc bạn train)
os.environ['JAVA_HOME'] = r'C:\Program Files\Java\jdk-11.0.30'
os.environ['HADOOP_HOME'] = r'D:\Hadoop'
os.environ['SPARK_LOCAL_IP'] = '127.0.0.1'

os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

# 2. Khởi động Spark và Nạp mô hình (Chỉ nạp 1 lần khi bật Server)
print("Đang khởi động Máy chủ Spark AI...")
spark = SparkSession.builder.appName("DiabetesAPI").getOrCreate()

current_dir = os.getcwd().replace("\\", "/")
MODEL_DIR = f"file:///{current_dir}/spark_models_diabetes_readmission"

print("Đang nạp mô hình tiền xử lý...")
preprocess_pipeline = PipelineModel.load(f"{MODEL_DIR}/preprocess_model")

print("Đang nạp mô hình Random Forest...")
rf_model = RandomForestClassificationModel.load(f"{MODEL_DIR}/best_model_Random_Forest")
print("✅ Hệ thống đã sẵn sàng phục vụ!")

# 3. Khởi tạo Web API
app = FastAPI()

class PatientData(BaseModel):
    age_group: float
    gender: str
    race: str
    time_in_hospital: float
    num_lab_procedures: float
    discharge_disposition_id_cat: str
    number_inpatient: float
    number_emergency: float
    number_outpatient: float
    number_diagnoses: float

@app.post("/predict")
async def predict_risk(data: PatientData):
    try:
        # Bước A: Map dữ liệu từ Web vào row_dict
        row_dict = {
            # 1. Các biến lấy TRỰC TIẾP TỪ GIAO DIỆN WEB
            "age_group": float(data.age_group), # Ép kiểu float cho chắc chắn
            "gender": data.gender,
            "race": data.race,
            "time_in_hospital": float(data.time_in_hospital),
            "num_lab_procedures": float(data.num_lab_procedures),
            "discharge_disposition_id_cat": data.discharge_disposition_id_cat,
            "number_inpatient": float(data.number_inpatient),
            "number_emergency": float(data.number_emergency),
            "number_outpatient": float(data.number_outpatient),
            "number_diagnoses": float(data.number_diagnoses),
            
            # 2. Các biến CÒN LẠI để mặc định (Như cũ)
            "num_medications": 15.0,
            "num_procedures": 0.0,
            "payer_code": "?",
            "medical_specialty_clean": "Missing",
            "max_glu_serum": "None",
            "A1Cresult": "None",
            "metformin": "No", "repaglinide": "No", "nateglinide": "No", "chlorpropamide": "No",
            "glimepiride": "No", "acetohexamide": "No", "glipizide": "No", "glyburide": "No",
            "tolbutamide": "No", "pioglitazone": "No", "rosiglitazone": "No", "acarbose": "No",
            "miglitol": "No", "troglitazone": "No", "tolazamide": "No", "examide": "No",
            "citoglipton": "No", "insulin": "No", "glyburide-metformin": "No",
            "glipizide-metformin": "No", "glimepiride-pioglitazone": "No",
            "metformin-rosiglitazone": "No", "metformin-pioglitazone": "No",
            "change": "No", "diabetesMed": "Yes",
            "diag_1": "250.0", "diag_2": "250.0", "diag_3": "250.0",
            "admission_type_id_cat": "1",
            "admission_source_id_cat": "7",

            "has_weight": 0
        }
        
        # Bước B: Chuyển dữ liệu thành Spark DataFrame
        input_df = spark.createDataFrame([Row(**row_dict)])
        
        # Bước C: Chạy qua 파ipeline tiền xử lý
        processed_df = preprocess_pipeline.transform(input_df)
        
        # Bước D: Dự đoán bằng Random Forest
        prediction_df = rf_model.transform(processed_df)
        
        # Lấy kết quả (Cột 'probability' chứa mảng xác suất [0.8, 0.2])
        prob = prediction_df.select("probability").first()[0]
        risk_score = round(prob[1] * 100, 2)
        is_high_risk = int(prediction_df.select("prediction").first()[0])
        
        return {"status": "success", "prediction": is_high_risk, "risk_score": risk_score}
        
    except Exception as e:
        return {"status": "error", "message": str(e)}

# Phục vụ file HTML giao diện
@app.get("/", response_class=HTMLResponse)
async def read_index():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()