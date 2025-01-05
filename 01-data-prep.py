# Databricks notebook source
# MAGIC %md
# MAGIC # Data Preparation
# MAGIC
# MAGIC We begin our project with preparing the data. During Item Onboarding scenarios, companies tend to receive multiple peieces of data in different formats. Some of the most common cases are CSVs full of text, that hold data about item properties such as color, description, material, etc.. as well as pictures of the item.
# MAGIC
# MAGIC In order to be able to simulate a similar scenario, we went looking for a similar dataset. [Amazon's Berkley Objects Dataset](https://amazon-berkeley-objects.s3.amazonaws.com/index.html) had exactly what we were looking for. It features data about items, which is not 100% consistent, just as you would expect in a real life scenario. And, it also features images which we can use with a vision model to extract information about the products.
# MAGIC
# MAGIC In this notebook, we will prepare the environment, download the data, unzip it and save it so we can use it in the later stages.
# MAGIC
# MAGIC The recommended compute here is a single noded simple machine with the latest runtime. We used a machine that had `4 CPU Cores` with `32 GB of RAM Memory`, and `Runtime 15.4 LTS`. You do not need a cluster or GPUs in this notebook.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Prepare Containers
# MAGIC
# MAGIC Here, we will start by leveraging [Unity Catalog](https://docs.databricks.com/en/data-governance/unity-catalog/index.html) to create some containers, as in Catalog, and a Schema (Database) which we will using for storing our tables.
# MAGIC
# MAGIC We will also create a [Volume](https://docs.databricks.com/en/sql/language-manual/sql-ref-volumes.html) within this Schema to store our files. You can think of Volumes as a hard-drive like storage location which works great for storing actual files like CSVs, or Images

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Using an existing catalog here by default in this notebook
# MAGIC USE CATALOG mas;
# MAGIC -- If a new catalog is needed: CREATE CATALOG IF NOT EXISTS xyz;
# MAGIC
# MAGIC -- Creating a schema within that catalog which will hold our tables
# MAGIC CREATE SCHEMA IF NOT EXISTS item_onboarding;
# MAGIC
# MAGIC -- Use this schema by default for all operations in this notebook
# MAGIC USE SCHEMA item_onboarding;
# MAGIC
# MAGIC -- Create a volume to hold the files
# MAGIC CREATE VOLUME IF NOT EXISTS landing_zone;

# COMMAND ----------

# MAGIC %md
# MAGIC ### Download Data
# MAGIC
# MAGIC In this section, we will use some shell scripting to download the data and store it in the volume we just created

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Download Tabular Data
# MAGIC
# MAGIC We begin with the tabular data. The shell script also includes a part that unzips the downloaded data. This is needed before we can read it with Spark

# COMMAND ----------

# MAGIC %sh
# MAGIC
# MAGIC # Move to the target Volume directory
# MAGIC cd /Volumes/mas/item_onboarding/landing_zone
# MAGIC
# MAGIC # Download the listings files
# MAGIC echo "Downloading listings"
# MAGIC wget -q https://amazon-berkeley-objects.s3.amazonaws.com/archives/abo-listings.tar
# MAGIC
# MAGIC # Decompress the listings files
# MAGIC echo "Unzipping listings"
# MAGIC tar -xf ./abo-listings.tar --no-same-owner
# MAGIC gunzip ./listings/metadata/*.gz
# MAGIC
# MAGIC echo "Completed"

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Download Images
# MAGIC
# MAGIC Downloading the images is a little bit different, we follow some of the same procedures, however the moving to the volumes part is going to be different. Also, we do not download the data directly to the Volume, however we use the temporary memory of the Spark Driver here to execute the operation. 
# MAGIC
# MAGIC That is because it is faster to unzip in memory rather than a volume location when you have many small files, like images. 

# COMMAND ----------

# MAGIC %sh
# MAGIC
# MAGIC # Create a temp directory 
# MAGIC mkdir /tmp_landing_zone
# MAGIC
# MAGIC # Move to the target directory
# MAGIC cd /tmp_landing_zone
# MAGIC
# MAGIC # download the images files
# MAGIC echo "Downloading images"
# MAGIC wget -q https://amazon-berkeley-objects.s3.amazonaws.com/archives/abo-images-small.tar
# MAGIC
# MAGIC # Decompress the images files
# MAGIC # (untars to a folder called images)
# MAGIC echo "Unzipping images"
# MAGIC tar -xf ./abo-images-small.tar --no-same-owner
# MAGIC gzip -df ./images/metadata/images.csv.gz
# MAGIC
# MAGIC echo "Completed"

# COMMAND ----------

# MAGIC %md
# MAGIC **Image Copy Trick**
# MAGIC
# MAGIC The regular Databricks Utility to copy files around works great when you have few large files, however it is not as fast when you have many many small files like we do here. This can occur in scenarios where you work with images. For that reason, we produce a small utility to do threaded copy for us. 
# MAGIC
# MAGIC This utility will be used here to copy the images we unzipped from the driver's memory to the volume path we specify. It will work about 150x faster than how it would be if you were using the regular version

# COMMAND ----------

# Standard Imports
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

# External Imports
from tqdm import tqdm


# TODO: Check the number of optimal threads
def threaded_dbutils_copy(source_directory, target_directory, n_threads=10):
  """
  Copy source directory to target directory with threads.
  
  This function uses threads to execute multiple copy commands to speed up
  the copy process. Especially useful when dealing with multiple small files
  like images.
  
  :param source_directory: directory where the files are going to be copied from
  :param target_directory: directory where the files are going to be copied to
  :param n_threads: number of threads to use, bigger the number, faster the process
  
  Notes
    - Do not include backslashes at the end of the paths.
    - Increasing n_threads will put more load on the driver, keep an eye on the metrics
    to make sure the driver doesn't get overloaded
    - 100 threads pushes a decent driver properly
  """
  
  print("Listing all the paths")
  
  # Creating an empty list for all fiels
  all_files = []
  
  # Recursive search function for discovering all the files
  # TODO: Turn this into a generator
  def recursive_search(_path):
    file_paths = dbutils.fs.ls(_path)
    for file_path in file_paths:
      if file_path.isFile():
        all_files.append(file_path.path)
      else:
        recursive_search(file_path.path)
  
  # Applying recursive search to source directory
  recursive_search(source_directory)
  
  # Formatting path strings
  all_files = [path.split(source_directory)[-1][1:] for path in all_files]
  
  n_files = len(all_files)
  print(f"{n_files} files found")
  print(f"Beginning copy with {n_threads} threads")
  
  # Initiating TQDM with a thread lock for building a progress bar 
  p_bar = tqdm(total=n_files, unit=" copies")
  bar_lock = Lock()
  
  # Defining the work to be executed by a single thread
  def single_thread_copy(file_sub_path):
    dbutils.fs.cp(f"{source_directory}/{file_sub_path}", f"{target_directory}/{file_sub_path}")
    with bar_lock:
      p_bar.update(1)
  
  # Mapping the thread work accross all paths 
  with ThreadPoolExecutor(max_workers=n_threads, thread_name_prefix="copy_thread") as ex:
    ex.map(single_thread_copy, all_files)
  
  # Closing the progress bar
  p_bar.close()
  
  print("Copy complete")
  return

# COMMAND ----------

# Specify the paths
source_dir = "file:/tmp_landing_zone"
target_dir = "/Volumes/mas/item_onboarding/landing_zone/"

# Execute the copy
threaded_dbutils_copy(
  source_directory=source_dir, 
  target_directory=target_dir, 
  n_threads=150 # How many threads do we want running concurrently ? Don't be afraid to push the number..
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Read Raw Data and Save
# MAGIC
# MAGIC Now that we have moved our Raw data to the Volume location, we can go ahead and read it and save it as a Delta Table.

# COMMAND ----------

# Read data
products_df = (
    spark.read.json("/Volumes/mas/item_onboarding/landing_zone/listings/metadata")
)

# Save the raw data overwrite schema
(
    products_df
    .write
    .mode("overwrite")
    .option("overwriteSchema", "True")
    .saveAsTable("mas.item_onboarding.products_raw")
)

# display(products_df)

# COMMAND ----------

# Import
from pyspark.sql import functions as SF

# Read data
image_meta_df = (
  spark
      .read
      .csv(
        path="/Volumes/mas/item_onboarding/landing_zone/images/metadata",
        sep=',',
        header=True
    ) 
)

# Save Images 
(
    image_meta_df
    .write
    .mode("overwrite")
    .option("overwriteSchema", "True")
    .saveAsTable("mas.item_onboarding.image_meta_raw")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Basic Cleaning
# MAGIC
# MAGIC The text based data has some nested parts. We will do some basic cleaning and extraction to turn it into a usable format

# COMMAND ----------

# Imports
from pyspark.sql import functions as SF

# Read data
products_df = spark.read.table("mas.item_onboarding.products_raw")


# Build function for extracting values out of the standard columns
def value_extractor(df, target_col, sep=""):
    df = (
        df
        .withColumn(
            target_col,
            SF.expr(
                f"""concat_ws('{sep} ', filter({target_col}, x -> x.language_tag in ("en_US")).value)"""
            ),
        )
    )
    return df


# Create a transformed dataframe focussed on US products
products_clean_df = products_df.filter(SF.col("country").isin(["US"]))

# Apply Transofmration
transformation_columns = [
    ("brand", ""),
    ("bullet_point", ""),
    ("color", ""),
    ("item_keywords", " |"),
    ("item_name", ""),
    ("material", " |"),
    ("model_name", ""),
    ("product_description", ""),
    ("style", ""),
    ("fabric_type", ""),
    ("finish_type", ""),
]

for row in transformation_columns:
    products_clean_df = value_extractor(products_clean_df, row[0], row[1])

# Specify meta columns
meta_columns = [
    ### Meta
    "item_id",
    "country",
    "main_image_id",
]

transformed_columns = []
for row in transformation_columns:
    transformed_columns.append(row[0])

in_place_transformed_columns = [
    ### In place transform
    "product_type.value[0] AS product_type",
    "node.node_name[0] AS node_name",
]


# Apply column transformations and selections
products_clean_df = products_clean_df.selectExpr(
    meta_columns + transformed_columns + in_place_transformed_columns
)

# Drop duplicates based on item_id
products_clean_df = products_clean_df.dropDuplicates(["item_id"])

# Save cleaed products
(
    products_clean_df.write.mode("overwrite")
    .option("overwriteSchema", "True")
    .saveAsTable("mas.item_onboarding.products_clean")
)


# COMMAND ----------

# MAGIC %md
# MAGIC ### Image Meta Enrichment
# MAGIC
# MAGIC We then move on the enriching the image meta data with paths of the images, so we can have an easier time matching products with the paths of the main image ids later on.

# COMMAND ----------

from pyspark.sql import functions as SF

# Read DFs
products_clean_df = spark.read.table("mas.item_onboarding.products_clean")
image_meta_df = spark.read.table("mas.item_onboarding.image_meta_raw")

# Enrich with main image id
image_meta_enriched_df = image_meta_df.join(
    products_clean_df.selectExpr("main_image_id AS image_id", "item_id"),
    on="image_id",
    how="left",
)


# Build real path
real_path_prefix = "/Volumes/mas/item_onboarding/landing_zone/images/small/"
image_meta_enriched_df = image_meta_enriched_df.withColumn(
    "real_path", 
    SF.concat(
        SF.lit(real_path_prefix),  # Convert string to literal
        SF.col('path')
    )
)

# Save
(
    image_meta_enriched_df.write.mode("overwrite")
    .option("overwriteSchema", "True")
    .saveAsTable("mas.item_onboarding.image_meta_enriched")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Sample and Create Test Data
# MAGIC
# MAGIC For the sake of speed and reproducibility, we choose to focus on 100 items. This will help us process a batch of data in a timely fashion and will also help with re-producing results. However, if you would like to run project at a larger scale, feel free to change the limiting number from 100 to something greater, or comment out the limit statement all together to run it at full scale.

# COMMAND ----------

# Get a limited number of products for testing
sample_df = (
    spark.read.table("mas.item_onboarding.products_clean")
    .select("item_id")
    .limit(100) # Increase or comment out as you see fit.
)

# Save the limited number of products for testing
(
    sample_df
    .write
    .mode("overwrite")
    .option("overwriteSchema", "True")
    .saveAsTable("mas.item_onboarding.sample")
)

# COMMAND ----------

# Sample Products Clean
products_clean_df = spark.read.table("mas.item_onboarding.products_clean")
sample_df = spark.read.table("mas.item_onboarding.sample")
sampled_products_clean_df = sample_df.join(products_clean_df, on="item_id", how="left")

# Save
(
    sampled_products_clean_df
    .write
    .mode("overwrite")
    .option("overwriteSchema", "True")
    .saveAsTable("mas.item_onboarding.products_clean_sampled")
)

# COMMAND ----------

# Sample Images 
image_meta_enriched_df = spark.read.table("mas.item_onboarding.image_meta_enriched")
sample_df = spark.read.table("mas.item_onboarding.sample")
sampled_image_meta_enriched_df = sample_df.join(image_meta_enriched_df, on="item_id", how="left")

# Save
(
    sampled_image_meta_enriched_df
    .write
    .mode("overwrite")
    .option("overwriteSchema", "True")
    .saveAsTable("mas.item_onboarding.image_meta_enriched_sampled")
)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC This sums up the data prepartion notebook. In the next notebooks, we will use the sampled tables as well as the images we saved in the Volume to begin with information extraction
