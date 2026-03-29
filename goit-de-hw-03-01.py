import os
import sys
import subprocess

os.environ["JAVA_HOME"] = "/Library/Java/JavaVirtualMachines/temurin-17.jdk/Contents/Home"
os.environ["PATH"] = os.environ["JAVA_HOME"] + "/bin:" + os.environ["PATH"]
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

print("PYTHON:", sys.executable)
print("JAVA_HOME:", os.environ["JAVA_HOME"])
subprocess.run(["java", "-version"])

from pyspark.sql import SparkSession
from pyspark.sql.functions import *

# Spark session
spark = SparkSession.builder.appName("HM_DE_1").getOrCreate()

# Path settings for dataset
base_dir = os.path.dirname(os.path.abspath(__file__))

users_path = os.path.join(base_dir, "users.csv")
products_path = os.path.join(base_dir, "products.csv")
purchases_path = os.path.join(base_dir, "purchases.csv")

# Load dataset
users_df = spark.read.csv(users_path, header=True, inferSchema=True)
products_df = spark.read.csv(products_path, header=True, inferSchema=True)
purchases_df = spark.read.csv(purchases_path, header=True, inferSchema=True)

# Show first 10 rows
users_df.show(10)
products_df.show(10)
purchases_df.show(10)

# Remove rows with missing values
users_df = users_df.dropna()
products_df = products_df.dropna()
purchases_df = purchases_df.dropna()

# Total purchase quantity by product category
category_purchases_df = purchases_df.join(products_df, "product_id") \
    .select("category", "quantity") \
    .groupBy("category") \
    .agg(sum("quantity").alias("purchases_by_category"))

category_purchases_df.show()

# Total purchase quantity by category for users age 18-25
category_purchases_18_25_df = purchases_df.join(products_df, "product_id") \
    .join(users_df, "user_id") \
    .where((col("age") >= 18) & (col("age") <= 25)) \
    .select("category", "quantity", "age") \
    .groupBy("category") \
    .agg(sum("quantity").alias("purchases_by_category_18_25"))

category_purchases_18_25_df.show()


# Share of purchases quantity by category for users age 18-25
share_of_category_purchases_18_25_df = purchases_df.join(products_df, "product_id") \
    .join(users_df, "user_id") \
    .where((col("age") >= 18) & (col("age") <= 25)) \
    .select("category", "quantity", "price", "age") \
    .groupBy("category") \
    .agg((round(sum("quantity") / sum(col("price") * col("quantity")),2)).alias("share_of_purchases_by_category_18_25"))

share_of_category_purchases_18_25_df.show()

# Top 3 share of purchases quantity by category for users age 18-25
base_df = purchases_df.join(products_df, "product_id") \
    .join(users_df, "user_id") \
    .where((col("age") >= 18) & (col("age") <= 25)) \
    .withColumn("total_spent", col("price") * col("quantity"))

total_spent_all = base_df.agg(sum("total_spent").alias("total")).collect()[0]["total"]

result_df = base_df.groupBy("category") \
    .agg(round(sum("total_spent"),2).alias("category_spent")) \
    .withColumn(
        "percentage",
        round(col("category_spent") / total_spent_all * 100, 2)
    ) \
    .orderBy(col("percentage").desc()) \
    .limit(3)

result_df.show()

# Close spark session
spark.stop()

