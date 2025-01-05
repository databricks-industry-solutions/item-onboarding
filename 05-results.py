# Databricks notebook source
# MAGIC %md
# MAGIC # Results
# MAGIC
# MAGIC We will save the results as Delta tables given that we have generated Parquets during inference, and then we will build a simple interface to monitor the results.
# MAGIC
# MAGIC The recommended compute here is a single noded simple machine with the latest runtime. We used a machine that had `4 CPU Cores` with `32 GB of RAM Memory`, and `Runtime 15.4 LTS`. You do not need a cluster or GPUs in this notebook.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Save Image Analysis 
# MAGIC
# MAGIC Lets begin with our image analysis. We have the interim dataframe as a Parquet in the Volume directory. We can pick the Parquet files and save them as Delta in the Unity Catalog.

# COMMAND ----------

# read parquet file
from pyspark.sql import functions as SF

image_analysis_df = spark.read.parquet(
    "/Volumes/mas/item_onboarding/interim_data/image_analysis"
)

image_analysis_df = image_analysis_df.drop("image")

pattern = r"assistant<\|end_header_id\|>\s*([\s\S]*?)<\|eot_id\|>"
image_analysis_df = (
    image_analysis_df
    .withColumn("gen_description", SF.regexp_extract("description", pattern, 1))
    .withColumn("gen_color", SF.regexp_extract("color", pattern, 1))
)

(
    image_analysis_df
    .write
    .mode("overwrite")
    .option("overwriteSchema", "True")
    .saveAsTable("mas.item_onboarding.image_analysis")
)

display(image_analysis_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Save Text Analysis
# MAGIC
# MAGIC We will repeat the same process for the text analaysis part.

# COMMAND ----------

# Read the parquet file
text_analysis_df = spark.read.parquet("/Volumes/mas/item_onboarding/interim_data/results")

# Save a Delta table
(
    text_analysis_df
    .write
    .mode("overwrite")
    .option("overwriteSchema", "True")
    .saveAsTable("mas.item_onboarding.text_analysis")
)

display(text_analysis_df)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # Results Interface 
# MAGIC
# MAGIC Lets build an interface where we can easily monitor our process. We want to be able to select from a product ID, understand what was the data given to us, what the image model saw, and what the text model decided to build. 

# COMMAND ----------

text_analysis_df = spark.read.table("mas.item_onboarding.text_analysis")

# Get all the available IDs
available_ids = [x[0] for x in text_analysis_df.select("item_id").distinct().collect()]

# Select one
index = 2
selected_id = available_ids[index]

# Get item Data
item_data = text_analysis_df.filter(text_analysis_df.item_id == selected_id).collect()[0]

# COMMAND ----------

# MAGIC %md
# MAGIC ### Check Out the Item
# MAGIC
# MAGIC Beginning by viewing the item's image.

# COMMAND ----------

from PIL import Image
print(f">>> Beginning Analysis for Item ID: {item_data['item_id']} <<<\n")
print("Item's image is as follows:")
img = Image.open(item_data["real_path"])
display(img)

# COMMAND ----------

# MAGIC %md
# MAGIC ### The Information from the Supplier
# MAGIC
# MAGIC This part shows the data supplier gives us for the item.

# COMMAND ----------

print(f">>> Item Description: \n\n{item_data['bullet_point']}")
print(f"\n\n>>> Item Color: \n\n{item_data['color']}")
print(f"\n\n>>> Item Keywords: \n\n{item_data['item_keywords']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Image Model Analysis
# MAGIC
# MAGIC What the image model saw ?

# COMMAND ----------

print(f">>> What do you see in the image ?: \n\n{item_data['gen_description']}")
print(f"\n\n>>> What color is the product ?: \n\n{item_data['gen_color']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Text Model Analysis
# MAGIC
# MAGIC What the text model decided the suggest considering all data points

# COMMAND ----------

print(f">>> Suggested Description: \n\n{item_data['suggested_description'].strip()}")
print(f"\n\n>>> Suggested Color: \n\n{item_data['suggested_color'].strip()}")
print(f"\n\n>>> Suggested Keywords: \n\n{item_data['suggested_keywords'].strip()}")
print(f"\n>>> Suggested Category: \n\n{item_data['suggested_category'].strip()}")

# COMMAND ----------


