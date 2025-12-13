#!/usr/bin/env python
# coding: utf-8

# In[94]:


from pyspark.sql import SparkSession


spark = (
    SparkSession.builder.appName("PurchaseIntentionAnalysis")
    .master("spark://spark-master:7077")
    .config("spark.sql.ansi.enabled", "false")
    .config("spark.sql.repl.eagerEval.enabled", "true")
    .getOrCreate()
)

sessions_data = spark.read.csv(["/opt/spark/data/worker1/*.csv"], header=True, inferSchema=True)
sessions_data.createOrReplaceTempView("sessions_data")
sessions_data.repartition(3)

sessions_data


# In[68]:


from pyspark.sql.functions import udf, col
from pyspark.sql.types import FloatType, IntegerType, BooleanType, StringType

@udf(FloatType())
def ratio_duration_per_visit(visitc, duration):
  return 0 if visitc == 0 else duration / visitc
  
@udf(BooleanType())
def is_special_date(special_day):
  return special_day > 0

@udf(StringType())
def operating_system_label(os):
  match os:
    case 1:
      return 'OS_1'
    case 2:
      return 'OS_2'
    case 3:
      return 'OS_3'
    case _:
      return 'OS_Other'
    
@udf(StringType())
def region_label(region):
  return 'Region_1' if region == 1 else 'Region_Other'

@udf(StringType())
def traffic_type_label(traffic_type):
  return 'TrafficType_1_3' if 1 <= traffic_type <= 3 else 'TrafficType_Other'

@udf(IntegerType())
def month_number(month):
  month_mapping = {
    'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'June': 6,
    'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
  }
  return month_mapping[month]
sessions_data_fe = sessions_data.withColumns({
  "Administrative_Duration_Per_Visit": ratio_duration_per_visit(
    col("Administrative"), col("Administrative_Duration")
  ),
  "Informational_Duration_Per_Visit": ratio_duration_per_visit(
    col("Informational"), col("Informational_Duration")
  ),
  "ProductRelated_Duration_Per_Visit": ratio_duration_per_visit(
    col("ProductRelated"), col("ProductRelated_Duration")
  ),
  "Is_Special_Date": is_special_date(col("SpecialDay")),
  "OperatingSystems": operating_system_label(col("OperatingSystems")),
  "Region": region_label(col("Region")),
  "TrafficType": traffic_type_label(col("TrafficType")),
  "Month_Number": month_number(col("Month"))
})
sessions_data_fe


# In[69]:


# only important features
sessions_data_fee = sessions_data_fe.select([
  "Revenue",
  "Administrative_Duration_Per_Visit",
  "Informational_Duration_Per_Visit",
  "ProductRelated_Duration_Per_Visit",
  "BounceRates",
  "ExitRates",
  "PageValues",
  "Is_Special_Date",
  "Month_Number",
  "OperatingSystems",
  "Region",
  "TrafficType",
  "Weekend",
  "VisitorType"
])
sessions_data_fee


# ## Handle 0 values

# In[70]:


from pyspark.sql.functions import sum as spark_sum, col, coalesce, lit


# In[71]:


# Filter rows where at least one column is null

def print_null_rows(df):
    rows_with_nulls = df.filter(
        " OR ".join([f"`{c}` IS NULL" for c in df.columns]) # if any row is null in any column
    )

    print(f"Total rows with null values: {rows_with_nulls.count()}")
    rows_with_nulls.show(truncate=False)


def check_null_values(df):
    """Check and display null value counts per column"""
    # Get null counts per column
    null_counts = df.select([
        spark_sum(col(c).isNull().cast("int")).alias(c) 
        for c in df.columns
    ])
    
    print("Null value counts per column:")
    null_counts.show(vertical=True)
    
    # Show only columns with nulls
    row = null_counts.collect()[0]
    print("\nColumns with null values:")
    has_nulls = False
    for col_name in df.columns:
        count = row[col_name]
        if count > 0:
            print(f"  {col_name}: {count}")
            has_nulls = True
    
    if not has_nulls:
        print("  No null values found")
    
    return row


# In[72]:


check_null_values(sessions_data_fee)
print_null_rows(sessions_data_fee)


# In[73]:


# The Null rows should be 
sessions_data_fee = sessions_data_fee.select([
    coalesce(col(c), lit(0)).alias(c) if c in ['Administrative_Duration_Per_Visit', 'Informational_Duration_Per_Visit', 'ProductRelated_Duration_Per_Visit'] else col(c)
    for c in sessions_data_fee.columns
])


# In[74]:


print_null_rows(sessions_data_fee)


# In[75]:


sessions_data_fee


# # Feature Transformation Pipeline

# In[76]:


from pyspark.ml.feature import VectorAssembler, StandardScaler, StringIndexer, OneHotEncoder
from pyspark.ml import Pipeline

# Define numerical features
numerical_features = [
    'Administrative_Duration_Per_Visit', 'Informational_Duration_Per_Visit', 'ProductRelated_Duration_Per_Visit',
    'BounceRates', 'ExitRates', 'PageValues', 'Month_Number',                      
]

# Define categorical features to encode
categorical_features = ['VisitorType', 'OperatingSystems', 'Region', 'TrafficType']

# Convert boolean to integer (True → 1, False → 0)
sessions_data_fee = sessions_data_fee.withColumn(
    "Weekend", 
    col("Weekend").cast("int")
)

sessions_data_fee = sessions_data_fee.withColumn(
    "Is_Special_Date", 
    col("Is_Special_Date").cast("int")
)

sessions_data_fee = sessions_data_fee.withColumn(
    "Revenue", 
    col("Revenue").cast("int")
)

# Create StringIndexer and OneHotEncoder for categorical features
indexers = [
    StringIndexer(inputCol=col_name, outputCol=col_name + "_index", handleInvalid="keep")
    for col_name in categorical_features
]

encoders = [
    OneHotEncoder(inputCol=col_name + "_index", outputCol=col_name + "_encoded")
    for col_name in categorical_features
]

# Assemble numerical features
numerical_assembler = VectorAssembler(
    inputCols=numerical_features,
    outputCol="numerical_features"
)

# Scale numerical features
scaler = StandardScaler(
    inputCol="numerical_features",
    outputCol="scaled_numerical_features",
    withStd=True,
    withMean=True
)

# Combine all features
all_feature_cols = ["scaled_numerical_features"] + [f"{col}_encoded" for col in categorical_features]

final_assembler = VectorAssembler(
    inputCols=all_feature_cols,
    outputCol="features"
)

print("Feature transformation pipeline created successfully!")


# # Train-Test Split

# In[77]:


# Split data into training and testing sets (80/20 split)
train_data, test_data = sessions_data_fee.randomSplit([0.8, 0.2], seed=42)

print(f"Training set size: {train_data.count()}")
print(f"Test set size: {test_data.count()}")

# Check class distribution in training set
print("\nClass distribution in training set:")
train_data.groupBy("Revenue").count().show()


# # Model Training - Decision Tree

# In[78]:


LABEL_COL = "Revenue"


# In[79]:


spark.catalog.clearCache()


# In[80]:


from pyspark.ml.classification import DecisionTreeClassifier
# Create Decision Tree model
dt = DecisionTreeClassifier(
    featuresCol="features",
    labelCol=LABEL_COL,
    maxDepth=10,
    minInstancesPerNode=20
)

# Build pipeline
dt_pipeline = Pipeline(stages=indexers + encoders + [numerical_assembler, scaler, final_assembler, dt])

# Train the model
print("Training Decision Tree model...")
dt_model = dt_pipeline.fit(train_data)
print("Training complete!")

# Make predictions
dt_predictions = dt_model.transform(test_data)
dt_predictions.select(LABEL_COL, "prediction", "probability").show(20, truncate=False)


# # Model Evaluation - Decision Tree

# In[81]:


from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.sql.functions import count

# Binary classification metrics
binary_evaluator = BinaryClassificationEvaluator(
    labelCol=LABEL_COL,
    rawPredictionCol="rawPrediction",
    metricName="areaUnderROC"
)

dt_auc_roc = binary_evaluator.evaluate(dt_predictions)
print(f"Decision Tree - AUC-ROC: {dt_auc_roc:.4f}")

binary_evaluator.setMetricName("areaUnderPR")
dt_auc_pr = binary_evaluator.evaluate(dt_predictions)
print(f"Decision Tree - AUC-PR: {dt_auc_pr:.4f}")

# Multiclass metrics
multiclass_evaluator = MulticlassClassificationEvaluator(labelCol=LABEL_COL, predictionCol="prediction")

dt_accuracy = multiclass_evaluator.evaluate(dt_predictions, {multiclass_evaluator.metricName: "accuracy"})
dt_precision = multiclass_evaluator.evaluate(dt_predictions, {multiclass_evaluator.metricName: "weightedPrecision"})
dt_recall = multiclass_evaluator.evaluate(dt_predictions, {multiclass_evaluator.metricName: "weightedRecall"})
dt_f1 = multiclass_evaluator.evaluate(dt_predictions, {multiclass_evaluator.metricName: "f1"})

print(f"\nDecision Tree Metrics:")
print(f"Accuracy:  {dt_accuracy:.4f}")
print(f"Precision: {dt_precision:.4f}")
print(f"Recall:    {dt_recall:.4f}")
print(f"F1-Score:  {dt_f1:.4f}")

print("\nConfusion Matrix:")
dt_predictions.groupBy(LABEL_COL, "prediction").agg(count("*").alias("count")).orderBy(LABEL_COL, "prediction").show()

