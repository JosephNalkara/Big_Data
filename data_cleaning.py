from pyspark.sql import SparkSession
from pyspark.sql.functions import avg, pow, col, when, count, isnan
from pyspark.ml.feature import MinMaxScaler, VectorAssembler
from pyspark.ml import Pipeline

# Step 1: Create Spark session
spark = SparkSession.builder.appName("data cleaning").getOrCreate()

# Step 2: Input data
data = [
    ("john", 25, "new york", "john@example.com"),
    ("Alice", None, "Los Angeles", "alice@example.com"),
    ("BOB", 30, None, "bob@example.com"),
    (None, 28, "Chicago", "noemail@example.com"),
    ("john", 25, "new york", "john@example.com"),
    ("  Eve", 22, " boston ", "eve@example"),
    ("Mallory", 35, "Dallas", None),
    ("ALICE", 29, "Los Angeles", "alice@example.com")
]
columns = ["Name", "Age", "City", "Email"]
df = spark.createDataFrame(data, columns)

# Step 3: Show schema and raw data
print("SCHEMA:")
df.printSchema()
df.show()

# Step 4: Fill nulls
df_nodupli = df.fillna({
    "Name": "Unknown",
    "Age": 0,
    "City": "Unknown",
    "Email": "noemail@unknown.com"
})
print("✅ After filling nulls:")
df_nodupli.show()

# Step 5: Drop rows with any null values
df_nonull = df.dropna()
print("✅ After dropping rows with nulls:")
df_nonull.show()

# Step 6: Average age
df_avg_age = df.select(avg("Age").alias("Average_Age"))
print("📊 Average Age:")
df_avg_age.show()

# Step 7: Outliers (Age > 28)
print("🚨 Outliers (Age > 28):")
outliers = df.filter(df["Age"] > 28)
outliers.show()

# Step 8: Square transform of Age
df_squared = df.withColumn("Age_Squared", pow("Age", 2))
print("🧮 With Age Squared:")
df_squared.show()

# Step 9: Min-Max Scaling on Age (exclude nulls)
df_for_scaling = df.filter(df["Age"].isNotNull())

# Assemble into vector
assembler = VectorAssembler(inputCols=["Age"], outputCol="Age_Vector")
scaler = MinMaxScaler(inputCol="Age_Vector", outputCol="Age_Scaled")
pipeline = Pipeline(stages=[assembler, scaler])

scaler_model = pipeline.fit(df_for_scaling)
df_scaled = scaler_model.transform(df_for_scaling)

# Final output with scaling
print("📈 Min-Max Scaled Age:")
df_scaled.select("Name", "Age", "Age_Scaled").show(truncate=False)
