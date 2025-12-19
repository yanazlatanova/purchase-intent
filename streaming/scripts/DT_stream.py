#!/usr/bin/env python3
"""
Streaming Decision Tree (simplified):
- Connects to stream-simulator:9999
- Reads CSV lines
- Minimal feature engineering
- Drops rows with nulls in essential columns
- Trains & evaluates Decision Tree per batch
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, split, when
from pyspark.sql.types import FloatType, IntegerType, StringType
from pyspark.sql.functions import udf

from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder
from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator

SOCKET_HOST = "stream-simulator"  # or "localhost"
SOCKET_PORT = 9999

LABEL_COL = "Revenue"
NUMERICAL_FEATURES = [
    "Administrative_Duration_Per_Visit",
    "ProductRelated_Duration_Per_Visit",
    "BounceRates",
    "ExitRates",
    "PageValues",
]
CATEGORICAL_FEATURES = ["VisitorType", "OperatingSystems", "Region", "TrafficType"]


def create_spark_session():
    spark = (
        SparkSession.builder
        .appName("DT_Streaming_Simple")
        .master("spark://spark-master:7077")
        .config("spark.executor.cores", "1")
        .config("spark.executor.memory", "512m")
        .config("spark.sql.shuffle.partitions", "2")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark


@udf(FloatType())
def ratio_duration_per_visit(visitc, duration):
    if visitc is None or duration is None:
        return 0.0
    try:
        v = float(visitc)
        d = float(duration)
    except Exception:
        return 0.0
    return 0.0 if v == 0 else d / v


@udf(StringType())
def os_label(os):
    try:
        v = int(os)
    except Exception:
        return "OS_Other"
    return {1: "OS_1", 2: "OS_2", 3: "OS_3"}.get(v, "OS_Other")


@udf(StringType())
def region_label(region):
    try:
        r = int(region)
    except Exception:
        return "Region_Other"
    return "Region_1" if r == 1 else "Region_Other"


@udf(StringType())
def traffic_label(tt):
    try:
        t = int(tt)
    except Exception:
        return "TrafficType_Other"
    return "TrafficType_1_3" if 1 <= t <= 3 else "TrafficType_Other"


def engineer_features(df_raw):
    # Cast numeric columns and derive only a few features
    df = df_raw.select(
        col("Administrative").cast("double").alias("Administrative"),
        col("Administrative_Duration").cast("double").alias("Administrative_Duration"),
        col("ProductRelated").cast("double").alias("ProductRelated"),
        col("ProductRelated_Duration").cast("double").alias("ProductRelated_Duration"),
        col("BounceRates").cast("double").alias("BounceRates"),
        col("ExitRates").cast("double").alias("ExitRates"),
        col("PageValues").cast("double").alias("PageValues"),
        col("OperatingSystems").cast("int").alias("OperatingSystems"),
        col("Region").cast("int").alias("Region"),
        col("TrafficType").cast("int").alias("TrafficType"),
        col("VisitorType").alias("VisitorType"),
        col("Weekend").alias("Weekend"),
        col("Revenue").alias("Revenue"),
    )

    df = df.withColumns(
        {
            "Administrative_Duration_Per_Visit": ratio_duration_per_visit(
                col("Administrative"), col("Administrative_Duration")
            ),
            "ProductRelated_Duration_Per_Visit": ratio_duration_per_visit(
                col("ProductRelated"), col("ProductRelated_Duration")
            ),
            "OperatingSystems_lbl": os_label(col("OperatingSystems")),
            "Region_lbl": region_label(col("Region")),
            "TrafficType_lbl": traffic_label(col("TrafficType")),
            "Weekend_int": when(col("Weekend") == "True", 1).otherwise(0),
            "Revenue_int": when(col("Revenue") == "True", 1).otherwise(0),
        }
    )

    df = df.select(
        col("Revenue_int").alias(LABEL_COL),
        "Administrative_Duration_Per_Visit",
        "ProductRelated_Duration_Per_Visit",
        "BounceRates",
        "ExitRates",
        "PageValues",
        col("OperatingSystems_lbl").alias("OperatingSystems"),
        col("Region_lbl").alias("Region"),
        col("TrafficType_lbl").alias("TrafficType"),
        col("Weekend_int").alias("Weekend"),
        "VisitorType",
    )

    return df


def drop_all_nulls(df):
    cols_to_check = [LABEL_COL] + NUMERICAL_FEATURES + CATEGORICAL_FEATURES + ["Weekend"]
    return df.dropna(subset=cols_to_check)


def build_pipeline():
    indexers = [
        StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep")
        for c in CATEGORICAL_FEATURES
    ]
    encoders = [
        OneHotEncoder(inputCol=f"{c}_idx", outputCol=f"{c}_vec")
        for c in CATEGORICAL_FEATURES
    ]

    num_assembler = VectorAssembler(
        inputCols=NUMERICAL_FEATURES, outputCol="num_features"
    )

    final_assembler = VectorAssembler(
        inputCols=["num_features"] + [f"{c}_vec" for c in CATEGORICAL_FEATURES],
        outputCol="features",
    )

    dt = DecisionTreeClassifier(
        featuresCol="features", labelCol=LABEL_COL, maxDepth=4, minInstancesPerNode=10
    )

    return Pipeline(stages=indexers + encoders + [num_assembler, final_assembler, dt])


def evaluate(preds, batch_id):
    # Binary metrics
    beval = BinaryClassificationEvaluator(
        labelCol=LABEL_COL,
        rawPredictionCol="rawPrediction",
        metricName="areaUnderROC",
    )
    auc_roc = beval.evaluate(preds)
    beval.setMetricName("areaUnderPR")
    auc_pr = beval.evaluate(preds)

    # Multiclass / classification metrics
    meval = MulticlassClassificationEvaluator(
        labelCol=LABEL_COL,
        predictionCol="prediction",
    )

    acc = meval.evaluate(preds, {meval.metricName: "accuracy"})
    precision = meval.evaluate(preds, {meval.metricName: "weightedPrecision"})
    recall = meval.evaluate(preds, {meval.metricName: "weightedRecall"})
    f1 = meval.evaluate(preds, {meval.metricName: "f1"})

    print(f"\n=== Metrics batch {batch_id} ===")
    print(f"AUC-ROC:  {auc_roc:.4f}")
    print(f"AUC-PR:   {auc_pr:.4f}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision:{precision:.4f}")
    print(f"Recall:   {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")


def main():
    spark = create_spark_session()

    print(f"Connecting to socket {SOCKET_HOST}:{SOCKET_PORT} ...")

    df_stream = (
        spark.readStream.format("socket")
        .option("host", SOCKET_HOST)
        .option("port", SOCKET_PORT)
        .load()
    )

    df_parsed = df_stream.select(split(col("value"), ",").alias("fields")).select(
        col("fields")[0].alias("Administrative"),
        col("fields")[1].alias("Administrative_Duration"),
        col("fields")[2].alias("Informational"),
        col("fields")[3].alias("Informational_Duration"),
        col("fields")[4].alias("ProductRelated"),
        col("fields")[5].alias("ProductRelated_Duration"),
        col("fields")[6].alias("BounceRates"),
        col("fields")[7].alias("ExitRates"),
        col("fields")[8].alias("PageValues"),
        col("fields")[9].alias("SpecialDay"),
        col("fields")[10].alias("Month"),
        col("fields")[11].alias("OperatingSystems"),
        col("fields")[12].alias("Browser"),
        col("fields")[13].alias("Region"),
        col("fields")[14].alias("TrafficType"),
        col("fields")[15].alias("VisitorType"),
        col("fields")[16].alias("Weekend"),
        col("fields")[17].alias("Revenue"),
    )

    df_fe = engineer_features(df_parsed)
    pipeline = build_pipeline()

    def train_on_batch(batch_df, batch_id):
        raw_count = batch_df.count()
        if raw_count == 0:
            print(f"\nBatch {batch_id}: 0 rows")
            return

        print(f"\nBatch {batch_id}: raw rows = {raw_count}")
        cleaned = drop_all_nulls(batch_df)
        clean_count = cleaned.count()
        print(f"Batch {batch_id}: after null removal = {clean_count}")

        if clean_count < 20:
            print(f"Batch {batch_id}: not enough rows after cleaning")
            return

        # Optional: limit rows to keep training cheap
        cleaned = cleaned.limit(200)

        train_df, test_df = cleaned.randomSplit([0.8, 0.2], seed=42)
        n_train = train_df.count()
        n_test = test_df.count()
        print(f"Batch {batch_id}: train={n_train}, test={n_test}")

        if n_train == 0 or n_test == 0:
            print(f"Batch {batch_id}: empty train/test")
            return

        model = pipeline.fit(train_df)
        preds = model.transform(test_df)
        evaluate(preds, batch_id)

    query = (
        df_fe.writeStream.foreachBatch(train_on_batch)
        .outputMode("append")
        .trigger(processingTime="30 seconds")
        .start()
    )

    print("Started streaming DT. Waiting for data...\n")
    query.awaitTermination()


if __name__ == "__main__":
    main()
