# Databricks notebook source
# MAGIC %md
# MAGIC # Download Model Weights
# MAGIC
# MAGIC We will be using Open Source LLAMA models which will work great for our use case. They are small enough to fit comfortably in the A100 GPUs, have great performance, and they can easily get the job done.
# MAGIC
# MAGIC The two LLAMA models we are going to leverage in the project are:
# MAGIC
# MAGIC - [LLAMA 3.2 11B Vision Model](https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct)
# MAGIC - [LLAMA 3.1 8B Instruct Model](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)
# MAGIC
# MAGIC The vision model will be used for extracting information from the item pictures, and the Insturct model will be used for our text based queries.
# MAGIC
# MAGIC The reason we choose to download the model weights here is because if we don't each time we want to run this worklow, we would have to download the weights. It doesn't take hours to download them, however if we can save a 5-10 minutes each time, it can add up to be significant saving in the long term, as it is much efficient to load the model weights from an existing location.
# MAGIC
# MAGIC We will use HuggingFace to download the models. LLAMA models requires a simple registration on the website. Once you have done that, you can [generate a token](https://huggingface.co/settings/tokens) which we will then use in the rest of the workflow.
# MAGIC
# MAGIC The huggingface package is already installed in our runtime, so we do not need to re-install it.
# MAGIC
# MAGIC Similar to to the data prep notebook, we can use a single node compute here. No need for a cluster or a GPU at this point. A machine that has `4 CPU` with `32 GB RAM Memory` running the `15.4 ML LTS` Runtime should be able to do the trick. **We do need the ML Runtime though**
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create Containers
# MAGIC
# MAGIC Just as we did in the data preparation stage, we create a Volume Location to save the model weights. In this case, we are calling the location models.

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Use this catalog by default
# MAGIC USE CATALOG mas;
# MAGIC
# MAGIC -- Use this schema by default
# MAGIC USE SCHEMA item_onboarding;
# MAGIC
# MAGIC -- Create a volume if it doesnt exist
# MAGIC CREATE VOLUME IF NOT EXISTS models;

# COMMAND ----------

# MAGIC %md
# MAGIC ### Download Image Model
# MAGIC
# MAGIC Now, we can write a shell script to download the image model weights. You can remove the --quiet flag if you wish to track the downloads progress

# COMMAND ----------

# MAGIC %sh
# MAGIC
# MAGIC # Export HF Token
# MAGIC export HF_TOKEN="YOUR TOKEN GOES HERE"
# MAGIC
# MAGIC
# MAGIC # Run download command
# MAGIC huggingface-cli \
# MAGIC   download \
# MAGIC   "meta-llama/Llama-3.2-11B-Vision-Instruct" \
# MAGIC   --local-dir "/Volumes/mas/item_onboarding/models/llama-32-11b-vision-instruct" \
# MAGIC   --exclude "original/*" \ # we do not need the consolidated weights in this folder
# MAGIC   --quiet # remove this if you want to want to track the model download progress

# COMMAND ----------

# MAGIC %md
# MAGIC ### Download Text Model
# MAGIC
# MAGIC Similar to the image model, we donwload the text model's weights as well.

# COMMAND ----------

# MAGIC %sh
# MAGIC
# MAGIC # Export HF Token
# MAGIC export HF_TOKEN="YOUR TOKEN GOES HERE"
# MAGIC
# MAGIC # Run download command
# MAGIC huggingface-cli \
# MAGIC   download \
# MAGIC   "meta-llama/Meta-Llama-3.1-8B-Instruct" \
# MAGIC   --local-dir "/Volumes/mas/item_onboarding/models/llama-31-8b-instruct" \
# MAGIC   --exclude "original/*" \ # we do not need the consolidated weights in this folder
# MAGIC   --quiet # remove this if you want to want to track the model download progress

# COMMAND ----------

# MAGIC %md
# MAGIC The models weights should be good to go at this point.. The download of both combined can take up to 30 mins depending on the internet connection..
