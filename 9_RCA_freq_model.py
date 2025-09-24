import os, logging, sys, torch
import pandas as pd
from BDA_Scripts.print_to_log import StreamToLogger
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.sql.functions import when, col
from pyspark.ml import Pipeline
from pyspark.ml.functions import vector_to_array
import torch.nn as nn
from torch.optim import AdamW
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from datetime import datetime

def set_spark_python_logs(app_name, log_base_dir):
    log_dir = os.path.join(log_base_dir, app_name)
    os.makedirs(log_dir, exist_ok=True)

    # SPARK LOGS
    log4j_path = os.path.join(log_dir, "log4j.properties")
    with open(log4j_path, "w") as f:
        f.write(f"""
                       log4j.rootCategory=INFO, FILE
                       log4j.appender.FILE=org.apache.log4j.RollingFileAppender
                       log4j.appender.FILE.File={log_dir}/spark.log
                       log4j.appender.FILE.MaxFileSize=10MB
                       log4j.appender.FILE.MaxBackupIndex=5
                       log4j.appender.FILE.layout=org.apache.log4j.PatternLayout
                       log4j.appender.FILE.layout.ConversionPattern=%d{{yy/MM/dd HH:mm:ss}} %p %c{{1}}: %m%n
                       """)

    # PYTHON LOGS
    log4p_path = os.path.join(log_dir, "python.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        handlers=[logging.FileHandler(log4p_path)])
    logger = logging.getLogger(__name__)
    sys.stdout = StreamToLogger(logger, logging.INFO)
    sys.stderr = StreamToLogger(logger, logging.ERROR)
    return log4j_path

def create_save_plot(loss_history, timestamp, output_dir):
    # PLOT CREATION
    plt.figure()
    plt.plot(loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("RCA Training Loss Curve")

    # PLOT SAVE
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"training_loss_{timestamp}.png")
    plt.savefig(output_path)

    print(f"Plot saved in: {output_path}")

# ======================================== PRE-PROCESSING
def normalization(df):

    # NEW FEATURES CREATION

    df = df.withColumn("avg_speed_motorway",
                       when(col("seconds_motorway") != 0,
                            col("meters_motorway") / col("seconds_motorway")).otherwise(0))

    df = df.withColumn("avg_speed_urban",
                       when(col("seconds_urban") != 0,
                            col("meters_urban") / col("seconds_urban")).otherwise(0))

    df = df.withColumn("avg_speed_other",
                       when(col("seconds_other") != 0,
                            col("meters_other") / col("seconds_other")).otherwise(0))

    df = df.withColumn("avg_speed_monday", when(col("seconds_travelled_monday") != 0,
                                                col("meters_travelled_monday") / col(
                                                    "seconds_travelled_monday")).otherwise(0))

    df = df.withColumn("avg_speed_tuesday", when(col("seconds_travelled_tuesday") != 0,
                                                 col("meters_travelled_tuesday") / col(
                                                     "seconds_travelled_tuesday")).otherwise(0))

    df = df.withColumn("avg_speed_wednesday", when(col("seconds_travelled_wednesday") != 0,
                                                   col("meters_travelled_wednesday") / col(
                                                       "seconds_travelled_wednesday")).otherwise(0))

    df = df.withColumn("avg_speed_thursday", when(col("seconds_travelled_thursday") != 0,
                                                  col("meters_travelled_thursday") / col(
                                                      "seconds_travelled_thursday")).otherwise(0))

    df = df.withColumn("avg_speed_friday", when(col("seconds_travelled_friday") != 0,
                                                col("meters_travelled_friday") / col(
                                                    "seconds_travelled_friday")).otherwise(0))

    df = df.withColumn("avg_speed_saturday", when(col("seconds_travelled_saturday") != 0,
                                                  col("meters_travelled_saturday") / col(
                                                      "seconds_travelled_saturday")).otherwise(0))

    df = df.withColumn("avg_speed_sunday", when(col("seconds_travelled_sunday") != 0,
                                                col("meters_travelled_sunday") / col(
                                                    "seconds_travelled_sunday")).otherwise(0))

    df = df.withColumn("acc_motorway",
                       when(col("total_meters") != 0,
                            col("acceleration") * (col("meters_motorway") / col("total_meters"))).otherwise(0))

    df = df.withColumn("brake_motorway",
                       when(col("total_meters") != 0,
                            col("brake") * (col("meters_motorway") / col("total_meters"))).otherwise(0))

    df = df.withColumn("avg_speed_night",
                       when(col("seconds_travelled_night") != 0,
                            col("meters_travelled_night") / col("seconds_travelled_night")).otherwise(0))

    df = df.withColumn("avg_duration",
                       when(col("number_trips") != 0,col("duration") / col("number_trips")).otherwise(0))

    df = df.withColumn("perc_curve",
                       when((col("meters_urban") + col("meters_other")) != 0,
                            col("cornering") / (col("meters_urban") + col("meters_other"))).otherwise(0))

    df = df.withColumn("perc_change_line",
                       when(col("meters_motorway") != 0,
                            col("lateral_movement") / col("meters_motorway")).otherwise(0))

    # GAUSSIAN_NORMALIZATION
    cols_to_scale = ['total_meters', 'total_seconds', 'speeding', 'acceleration', 'brake', 'cornering','lateral_movement',
                     'avg_speed_motorway', 'avg_speed_urban', 'avg_speed_other', 'avg_speed_monday', 'avg_speed_tuesday',
                     'avg_speed_wednesday', 'avg_speed_thursday', 'avg_speed_friday', 'avg_speed_saturday',
                     'avg_speed_sunday','brake_motorway','acc_motorway', 'avg_speed_night', 'avg_duration']

    assembler = VectorAssembler(inputCols=cols_to_scale, outputCol="features_vec")
    scaler = StandardScaler(inputCol="features_vec", outputCol="scaled_features", withMean=True, withStd=True)
    pipeline = Pipeline(stages=[assembler, scaler])
    scaler_model = pipeline.fit(df)
    df = scaler_model.transform(df)

    # NORMALIZED COLUMNS OVERWRITING
    df = df.withColumn("scaled_array", vector_to_array("scaled_features"))
    for i, col_name in enumerate(cols_to_scale):
        df = df.withColumn(col_name, col("scaled_array")[i])

    # REMOVE TEMPORARY COLUMNS
    df = df.drop("features_vec", "scaled_features", "scaled_array")

    # CLAIMS BINARIZATION
    binary_cols = [
        'risk_1', 'risk_2', 'risk_3', 'risk_4', 'risk_5', 'risk_6', 'body_injuries', 'property_damage']
    for c in binary_cols:
        df = df.withColumn(c, when(col(c) != 0, 1).otherwise(0))

    # CLEANING
    df = df.fillna(0)
    df = df.dropDuplicates()

    # CLAIMS WITH FAULT AGGREGATION ([cardC, cardD] --> RCA)
    df = df.withColumn("rca", when(col("cardD") != 0, col("cardD")).otherwise(col("rca")))
    df = df.withColumn("rca", when(col("cardC") != 0, col("cardC")).otherwise(col("rca")))
    df = df.withColumn("rca", when(col("rca") != 0, 1).otherwise(0))

    return df

#========================== X/Y, TRAIN/TEST SPLITTING
def set_train_test(pandas_df):
    # CREATE TRAIN DATASET
    df_train_zero = pandas_df[pandas_df['rca'] == 0].sample(n=7450, random_state=42)
    df_train_one = pandas_df[pandas_df['rca'] == 1].sample(n=7550, random_state=42)
    df_train = pd.concat([df_train_one, df_train_zero])

    # REMOVE THE ROWS USED IN TRAINING TEST FROM THE DATASET
    pandas_df = pandas_df.drop(df_train.index)

    # RESET INDEX
    pandas_df = pandas_df.reset_index(drop=True)

    # CREATE TEST DATASET
    df_test_zero = pandas_df[pandas_df['rca'] == 0].sample(n=1890, random_state=42)
    df_test_one = pandas_df[pandas_df['rca'] == 1].sample(n=1890, random_state=42)

    df_test = pd.concat([df_test_one, df_test_zero])

    # DOMAIN SET
    X_train = df_train.drop(columns=['atti_vandalici', 'cardC', 'cardD', 'casco',
                                   'collision', 'glass', 'natural_events', 'maindriver_injuries', 'fire_theft', 'rca',
                                   'incurred','meters_motorway', 'meters_urban', 'meters_other', 'meters_travelled_monday',
                                   'meters_travelled_tuesday', 'meters_travelled_wednesday', 'meters_travelled_thursday',
                                   'meters_travelled_friday', 'meters_travelled_saturday', 'meters_travelled_sunday',
                                   'meters_travelled_day', 'meters_travelled_night',
                                   'seconds_travelled_monday','seconds_travelled_tuesday',
                                   'seconds_travelled_wednesday', 'seconds_travelled_thursday','seconds_travelled_friday',
                                   'seconds_travelled_day', 'speeding', 'acceleration', 'brake',
                                   'seconds_motorway', 'seconds_urban', 'seconds_other',
                                   'seconds_travelled_night','seconds_travelled_saturday', 'seconds_travelled_sunday',
                                   'speeding','acceleration','brake','cornering','lateral_movement','duration',
                                   'number_trips', 'body_injuries','property_damage','body_and_property_damage',
                                   'ingestion_date', 'contract', 'total_seconds'])

    y_train = df_train['rca']

    X_test = df_test.drop(columns=['atti_vandalici', 'cardC', 'cardD', 'casco',
                                     'collision', 'glass', 'natural_events', 'maindriver_injuries', 'fire_theft', 'rca',
                                     'incurred', 'meters_motorway', 'meters_urban', 'meters_other',
                                     'meters_travelled_monday',
                                     'meters_travelled_tuesday', 'meters_travelled_wednesday',
                                     'meters_travelled_thursday',
                                     'meters_travelled_friday', 'meters_travelled_saturday', 'meters_travelled_sunday',
                                     'meters_travelled_day', 'meters_travelled_night',
                                     'seconds_travelled_monday', 'seconds_travelled_tuesday',
                                     'seconds_travelled_wednesday', 'seconds_travelled_thursday',
                                     'seconds_travelled_friday',
                                     'seconds_travelled_day', 'speeding', 'acceleration', 'brake',
                                     'seconds_motorway', 'seconds_urban', 'seconds_other',
                                     'seconds_travelled_night', 'seconds_travelled_saturday',
                                     'seconds_travelled_sunday',
                                     'speeding', 'acceleration', 'brake', 'cornering', 'lateral_movement', 'duration',
                                     'number_trips', 'body_injuries', 'property_damage', 'body_and_property_damage',
                                     'ingestion_date', 'contract','total_seconds'])

    y_test = df_test['rca']

    pd.set_option('display.max_columns', None)
    print(X_train.head())

    X_train = torch.tensor(X_train.values, dtype=torch.float32)
    y_train = torch.tensor(y_train.values, dtype=torch.float32)
    X_test = torch.tensor(X_test.values, dtype=torch.float32)
    y_test = torch.tensor(y_test.values, dtype=torch.float32)

    # ENSURE Y SHAPE = (n, 1)
    y_train = y_train.unsqueeze(1)
    y_test = y_test.unsqueeze(1)

    return X_train, X_test, y_train, y_test

#======================== DL MODEL STRUCTURE
# SKIPPING FUNCTION
class Residual_Block(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.GELU(),
            nn.Dropout(0.2)
        )

    def forward(self, x):
        return x + self.block(x)

# DL MODEL : 3 LAYERS + RESIDUAL_BLOCK
class RCA_DL_Model(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)

        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)

        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)

        self.residual = Residual_Block(128)  # Residual block

        self.fc4 = nn.Linear(128, 64)
        self.fc_out = nn.Linear(64, 1)

        # REGULARIZATION (TO AVOID OVERFITTING)
        self.dropout1 = nn.Dropout(0.4)
        self.dropout2 = nn.Dropout(0.3)
        self.dropout3 = nn.Dropout(0.2)

        # ACTIVATION FUNCTION
        self.act = nn.GELU()

    def forward(self, x):
        x = self.dropout1(self.act(self.bn1(self.fc1(x))))
        x = self.dropout2(self.act(self.bn2(self.fc2(x))))
        x = self.dropout3(self.act(self.bn3(self.fc3(x))))
        x = self.residual(x)
        x = self.act(self.fc4(x))
        return self.fc_out(x)

def train_rca_model(X_train, y_train, device, num_epochs):
    X_train = X_train.to(device)
    y_train = y_train.to(device)

    # ============================================================ MODEL TRAINING
    # BUILD MODEL
    model = RCA_DL_Model(X_train.shape[1]).to(device)

    # LOSS FUNCTION
    criterion = nn.BCEWithLogitsLoss()

    # OPTIMIZER
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)  # L2 Regularization

    # DATA FOR TRAINING PLOT PRINTING
    loss_history = []

    # TRAINING LOOP
    for epoch in range(num_epochs):
        model.train()

        optimizer.zero_grad()  # gradient reset
        outputs = model(X_train)  # collect outputs
        loss = criterion(outputs, y_train)  # loss application
        loss.backward()  # Backpropagation
        optimizer.step()  # weight updates

        loss_history.append(loss.item())

        if (epoch + 1) % 50 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}] - Loss: {loss.item():.4f}")

    # ============================================================ EVALUATION ON TRAINING SET
    model.eval()
    with torch.no_grad():
        train_outputs = model(X_train).squeeze()
        train_probs = torch.sigmoid(train_outputs)
        train_preds = (train_probs >= 0.5).int()

    y_train_np = y_train.cpu().numpy()
    y_pred_np = train_preds.cpu().numpy()

    cm = confusion_matrix(y_train_np, y_pred_np)
    print("Confusion matrix (training set):")
    print(cm)

    print("\nClassification Report (training set):")
    print(classification_report(y_train_np, y_pred_np, digits=4))
    return model, loss_history

def evaluate_model(model, X_test, y_test, output_dir):
    # ====================== VALIDATION
    X_test = X_test.to(device).float()
    y_test = y_test.to(device).int()

    threshold = 0.50

    model.eval()
    with torch.no_grad():
        logits = model(X_test)
        probs = torch.sigmoid(logits).squeeze()
        y_pred = (probs >= threshold).int()

    # SWITCH TO CPU
    y_pred_np = y_pred.cpu().numpy()
    y_true_np = y_test.cpu().numpy()

    # REPORT CREATION
    accuracy = (y_pred_np == y_true_np).mean()

    # .txt BUILDING
    report = []
    report.append(f"\nTest Accuracy (threshold={threshold}): {accuracy * 100:.2f}%")
    report.append(classification_report(y_true_np, y_pred_np, digits=4))
    report.append("Confusion matrix:\n" + str(confusion_matrix(y_true_np, y_pred_np)))
    report_text = "\n".join(report)

    file_path = os.path.join(output_dir, f"RCA_report_training_{timestamp}.txt")

    with open(file_path, "w") as f:
        f.write(report_text)

    print(f"Report saved in: {file_path}")

# ========================== LOGS
app_name = "RCA_freq_model"
log_base_dir = "/home/matteo/PycharmProjects/Tesi/BDA_Scripts/logs/9_RCA_freq_model"
log4j_path = set_spark_python_logs(app_name, log_base_dir)

#=========================================================== SPARK SESSION
spark = SparkSession.builder \
        .appName(app_name) \
        .config("spark.sql.parquet.mergeSchema", "true") \
        .config("spark.sql.parquet.compression.codec", "snappy") \
        .config("spark.sql.debug.maxToStringFields", 100) \
        .config("spark.sql.hive.convertMetastoreParquet", "true") \
        .config("spark.driver.extraJavaOptions", f"-Dlog4j.configuration=file:{log4j_path}") \
        .config("spark.executor.extraJavaOptions", f"-Dlog4j.configuration=file:{log4j_path}") \
    .getOrCreate()

df = spark.read.parquet("hdfs://localhost:9000/user/dr.who/ingestion_area/historical/gold")

df = normalization(df)
pandas_df = df.toPandas()

spark.stop()

X_train, X_test, y_train, y_test = set_train_test(pandas_df)
# MOVE THE WORKLOAD ON GPU (NVIDIA RTX 4060)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# TRAINING PARAMETERS
num_epochs = 1500

model, loss_history = train_rca_model(X_train, y_train, device, num_epochs)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = f"./model_validations/RCA_freq_{timestamp}"
create_save_plot(loss_history, timestamp, output_dir)

evaluate_model(model, X_test, y_test, output_dir)

# SAVE MODEL WEIGHTS
torch.save(model, f"./model_knowledge/RCA_freq/_{timestamp}_.pth")
print(f"Model saved in ./model_knowledge/RCA_freq/_{timestamp}_.pth")
torch.cuda.empty_cache()