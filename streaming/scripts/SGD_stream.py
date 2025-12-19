from pyspark import SparkContext
from pyspark.streaming import StreamingContext
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.classification import StreamingLogisticRegressionWithSGD
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.evaluation import BinaryClassificationMetrics, MulticlassMetrics  # [web:93][web:118]

SOCKET_HOST = "stream-simulator"  # or "localhost" in your docker network
SOCKET_PORT = 9999

# Feature vector layout:
# [Administrative_Duration_Per_Visit,
#  ProductRelated_Duration_Per_Visit,
#  BounceRates,
#  ExitRates,
#  PageValues,
#  OS_is_1,
#  OS_is_2,
#  OS_is_3,
#  Region_is_1,
#  Traffic_1_3,
#  Weekend_int]
NUM_FEATURES = 11


def safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default


def safe_int(x, default=0):
    try:
        return int(x)
    except Exception:
        return default


def parse_and_featurize(line):
    """
    Parse one CSV line from the stream and return LabeledPoint(label, features).
    Expected 18 columns in this order (same as stream_simulator.py):
    Administrative,Administrative_Duration,Informational,Informational_Duration,
    ProductRelated,ProductRelated_Duration,BounceRates,ExitRates,PageValues,
    SpecialDay,Month,OperatingSystems,Browser,Region,TrafficType,
    VisitorType,Weekend,Revenue
    """
    parts = line.split(",")
    if len(parts) < 18:
        return None

    administrative = safe_float(parts[0])
    administrative_dur = safe_float(parts[1])
    productrelated = safe_float(parts[4])
    productrelated_dur = safe_float(parts[5])
    bounce_rates = safe_float(parts[6])
    exit_rates = safe_float(parts[7])
    page_values = safe_float(parts[8])
    operating_systems = safe_int(parts[11])
    region = safe_int(parts[13])
    traffic_type = safe_int(parts[14])
    weekend = parts[16]
    revenue = parts[17]

    def ratio_duration_per_visit(v, d):
        return 0.0 if v == 0 else d / v

    admin_dur_per_visit = ratio_duration_per_visit(administrative, administrative_dur)
    prod_dur_per_visit = ratio_duration_per_visit(productrelated, productrelated_dur)

    os_is_1 = 1.0 if operating_systems == 1 else 0.0
    os_is_2 = 1.0 if operating_systems == 2 else 0.0
    os_is_3 = 1.0 if operating_systems == 3 else 0.0
    region_is_1 = 1.0 if region == 1 else 0.0
    traffic_1_3 = 1.0 if 1 <= traffic_type <= 3 else 0.0

    weekend_int = 1.0 if weekend == "True" else 0.0
    revenue_int = 1.0 if revenue == "True" else 0.0

    features = [
        admin_dur_per_visit,
        prod_dur_per_visit,
        bounce_rates,
        exit_rates,
        page_values,
        os_is_1,
        os_is_2,
        os_is_3,
        region_is_1,
        traffic_1_3,
        weekend_int,
    ]

    return LabeledPoint(revenue_int, Vectors.dense(features))


def main():
    sc = SparkContext(appName="OnlineLogReg_SGD_MLlib")
    sc.setLogLevel("WARN")

    # 5-second micro-batches
    ssc = StreamingContext(sc, batchDuration=5)

    print(f"Connecting to socket {SOCKET_HOST}:{SOCKET_PORT} ...")

    # DStream from socket
    lines = ssc.socketTextStream(SOCKET_HOST, SOCKET_PORT)
    print("Connected.", lines)
    # Parse -> LabeledPoint and drop malformed rows
    labeled_stream = lines.map(parse_and_featurize).filter(lambda lp: lp is not None)

    # Online logistic regression model with SGD
    model = StreamingLogisticRegressionWithSGD(
        stepSize=0.1,
        numIterations=50,
        miniBatchFraction=1.0,
    )
    model.setInitialWeights(Vectors.dense([0.0] * NUM_FEATURES))

    # Incremental updates on each batch
    model.trainOn(labeled_stream)

    def evaluate_batch(rdd):
        if rdd.isEmpty():
            return

        current_model = model.latestModel()

        # Index each record to join predictions and labels safely
        indexed = rdd.zipWithIndex().map(lambda pair: (pair[1], pair[0]))
        labels_by_idx = indexed.map(lambda x: (x[0], x[1].label))
        features_by_idx = indexed.map(lambda x: (x[0], x[1].features))

        features_rdd = features_by_idx.map(lambda x: x[1])

        # Scores for AUC and hard predictions
        scores_rdd = current_model.clearThreshold().predict(features_rdd)
        preds_rdd = current_model.predict(features_rdd)

        scores_by_idx = scores_rdd.zipWithIndex().map(lambda x: (x[1], x[0]))
        preds_by_idx = preds_rdd.zipWithIndex().map(lambda x: (x[1], x[0]))

        # (score, label) and (pred, label)
        score_and_labels = scores_by_idx.join(labels_by_idx).map(
            lambda x: (x[1][0], x[1][1])
        )
        pred_and_labels = preds_by_idx.join(labels_by_idx).map(
            lambda x: (x[1][0], x[1][1])
        )

        bin_metrics = BinaryClassificationMetrics(score_and_labels)  # [web:93][web:98]
        auc_roc = bin_metrics.areaUnderROC
        auc_pr = bin_metrics.areaUnderPR

        multi_metrics = MulticlassMetrics(pred_and_labels)  # [web:31][web:118]
        accuracy = multi_metrics.accuracy
        precision = multi_metrics.precision(1.0)
        recall = multi_metrics.recall(1.0)
        f1 = multi_metrics.fMeasure(1.0)

        print("\n=== Online LR SGD: batch metrics ===")
        print(f"Count:    {rdd.count()}")
        print(f"AUC-ROC:  {auc_roc:.4f}")
        print(f"AUC-PR:   {auc_pr:.4f}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision:{precision:.4f}")
        print(f"Recall:   {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")

        samples = pred_and_labels.take(5)
        if samples:
            print("Sample (prediction, label):")
            for pred, lbl in samples:
                print(f"pred={pred}, label={lbl}")

        bin_metrics.unpersist()

    # Attach evaluator
    labeled_stream.foreachRDD(evaluate_batch)

    print("Started StreamingLogisticRegressionWithSGD (online SGD) with metrics. Waiting for data...\n")
    ssc.start()
    ssc.awaitTermination()


if __name__ == "__main__":
    main()
