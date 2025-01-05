# Databricks notebook source
# MAGIC %md
# MAGIC # Image Analysis
# MAGIC
# MAGIC Now that we have the data and the models ready, we can go ahead and begin our image analysis. In this section, our primary goal is going to be to extract useful information from the product images. What we can tell from our data is, sometimes the descriptions of the items might not be clear, or suppliers might forget to provide certain information such as product color.
# MAGIC
# MAGIC Given that we have the images of the products, we can focus on building a flow which extracts this information from the pictures of the items by using the visual model we downloaded in the previous notebook. 
# MAGIC
# MAGIC The visual model works in a simple way, we provide the an image alongisde with a prompt, such as "describe the item in the image" and then it returns us text. 
# MAGIC
# MAGIC For this notebook, we use a machine that has a GPU attached. The `NVIDIA A100` GPU works very well with the model we are going to use here as it has enough GPU Memory (~80 GB) to run the model we would like to use here. On Azure, `NC24_ads` could be a good choice for compute. 
# MAGIC
# MAGIC We also use the `15.4 ML GPU` Runtime which has the necessary packages installed

# COMMAND ----------

# MAGIC %md
# MAGIC ### Setup
# MAGIC
# MAGIC Beginning with a basic setup process by upgrading the transformers library which is what we need to run the model.

# COMMAND ----------

# MAGIC %sh
# MAGIC pip install --upgrade transformers -q

# COMMAND ----------

# Restarting the python is important after the installation, and this code needs to run on a seperate cell after the installation
dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Read Images
# MAGIC
# MAGIC We have the image paths listed in a dataframe. We can basically use that to read the actual images. The following code reads the dataframe that has the image paths and builds a list out of that, which we later on use for reading the images.

# COMMAND ----------

# Retrieve image path table and build the list for all images
image_meta_df = spark.read.table("mas.item_onboarding.image_meta_enriched_sampled")
image_meta_df = image_meta_df.select("real_path")

# Collect and build list
image_paths = image_meta_df.collect()
image_paths = [x.real_path for x in image_paths if x.real_path]

# COMMAND ----------

# MAGIC %md
# MAGIC Lets check out what a single image looks like

# COMMAND ----------

from PIL import Image
img = Image.open(image_paths[0])
img

# COMMAND ----------

# MAGIC %md
# MAGIC ### Interactive Programmming
# MAGIC
# MAGIC We now want to design an interface which we can use for doing some interactive programming and prompting with model. In this part, we will begin using the [RAY](https://www.ray.io/) framework which helps us manage our workflows on the GPUs better. Ray is great at running GPU based workflows, and it runs very smoothly on the Databricks platform. 
# MAGIC
# MAGIC We will first use the Actor functionality of Ray, which will load the model to the GPU for us as an Actor. The nice thing about actors are you can call them at any time you like, which helps with interactive programming. They do not get unloaded from the GPUs unless you specify. 
# MAGIC
# MAGIC One recommendation here, if you have access to your compute's Web Terminal (through Databricks), you might find it really interesting to inspect your GPUs memory and utilisation as we go through this section. You can do that by openning the web terminal and typing the following shell commands:
# MAGIC
# MAGIC ```sh
# MAGIC apt-get update
# MAGIC apt install nvtop
# MAGIC nvtop
# MAGIC ```
# MAGIC
# MAGIC This runs a utility which helps you monitor your GPU in real time.

# COMMAND ----------

# Import the necessary libraries
from PIL import Image

from transformers import MllamaForConditionalGeneration, MllamaProcessor
import transformers

import ray
import torch

# Initiate Ray
ray.init(ignore_reinit_error=True)

# Specify the model path where the model is stored (our Volume directory)
model_path = "/Volumes/mas/item_onboarding/models/llama-32-11b-vision-instruct"


# Define the RAY actor

@ray.remote(num_gpus=1)
class LlamaVisionActor:
    def __init__(self, model_path: str):
        # Register model path
        self.model_path = model_path

        # Load config and model
        self.model = MllamaForConditionalGeneration.from_pretrained(
            model_path,
            device_map="cuda:0",
            torch_dtype=torch.float16,
        )
        self.processor = MllamaProcessor.from_pretrained(model_path)

        # Move model to device
        self.model.to("cuda:0")
        self.model.eval()

    def generate(self, prompt, batch, max_new_tokens=128):

        messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": prompt}
            ]}
        ]

        input_text = self.processor.apply_chat_template(messages, add_generation_prmpt=True)
        outputs = []
        for item in batch["image"]:
            image = Image.fromarray(item)
            inputs = self.processor(
                image,
                input_text,
                add_special_tokens=False,
                return_tensors="pt"
            ).to(self.model.device)
            output = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
            output = self.processor.decode(output[0])
            outputs.append(output)

        return outputs


vision_actor = LlamaVisionActor.remote(model_path)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Description Prompt
# MAGIC Now that we have the GPU loaded to the memory, we can go ahead and try some prompts. First, we will need to load some images using RAY. Following that, we will use a basic desription prompt to get the model to explain to us what it sees in the image. If we wanted to do some changes to our prompt, almost similar to prompt engineering, this could be a good interface for us to test that.
# MAGIC

# COMMAND ----------

# Read images
images_df = ray.data.read_images(
    image_paths,
    include_paths=True,
    mode="RGB"
)

# Create a batch of 10 images
test_batch = images_df.take_batch(10)

# COMMAND ----------

# Write a description promot
prompt = "Describe the product in the image"

# Use the actor to generate results
results = ray.get(
    vision_actor.generate.remote(
        prompt=prompt,
        batch=test_batch,
        max_new_tokens=256,
    )
)

# COMMAND ----------

# MAGIC %md
# MAGIC At this point we have our results, so lets go ahead and inspect them.

# COMMAND ----------

# Print the first result, remove the header and the end of turn token
print(results[0].split("<|start_header_id|>assistant<|end_header_id|>")[1].split("<|eot_id|>")[0].strip())

# Show the image as well
img = Image.fromarray(test_batch["image"][0])
img

# COMMAND ----------

# MAGIC %md
# MAGIC ### Color Prompt
# MAGIC
# MAGIC We can try a similar flow, however this time we can aim to extract the color of the product as some of the products in our dataset has the color field missing. 
# MAGIC
# MAGIC The code flow should be the same, with the change of the prompt.

# COMMAND ----------

# Write a description promot
prompt = "What is the color of the product ?"

# Use the actor to generate results
color_results = ray.get(
    vision_actor.generate.remote(
        prompt=prompt,
        batch=test_batch,
        max_new_tokens=32,
    )
)

# COMMAND ----------

# Print the first result, remove the header and the end of turn token
print(color_results[1].split("<|start_header_id|>assistant<|end_header_id|>")[1].split("<|eot_id|>")[0].strip())
print("\n")
# Show the image as well
img = Image.fromarray(test_batch["image"][1])
img

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC If we wanted to ask more questions or extract more information about the product images, this is where we could test those too. Now that our interactive work is done, we can go ahead and unload the GPU and the Actor by shutting down Ray. In the next section, we will begin to focus on the batch inference.

# COMMAND ----------

ray.shutdown()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Define Inference logic
# MAGIC
# MAGIC Now that we know more or less how the prompts work against the images, we can go ahead and define our batch inference logic. Using the actors could have been an applicable solution here to, however using the `map_batches` API from Ray gives us better control over our batch inference when we want to run things at scale. 
# MAGIC
# MAGIC The inference logic code looks very similar in parts to the Actor's code, however for the classes we design to run in batch mode, we need to build two methods within the class: `__init__` and `__call__`. 
# MAGIC
# MAGIC The `__init__` method will be more or less the same as the one of the actor here. It will be called once we run initiate our batch inference
# MAGIC
# MAGIC The `__call__` method is what gets called once the class is instantiated. 
# MAGIC
# MAGIC The following design abides by these rules and prepares the model for batch inference.

# COMMAND ----------

# Imports
from transformers import MllamaForConditionalGeneration, MllamaProcessor
import transformers
from PIL import Image
import torch
import ray


class LlamaVisionPredictor:
    def __init__(self, model_path: str):
        # Register model path
        self.model_path = model_path

        # Load config and model
        self.model = MllamaForConditionalGeneration.from_pretrained(
            model_path,
            device_map="cuda:0",
            torch_dtype=torch.float16,
        )
        self.processor = MllamaProcessor.from_pretrained(model_path)

        # Move model to device
        self.model.to("cuda:0")
        self.model.eval()

    def __call__(self, batch):
        # All inference logic goes here
        batch["description"] = self.generate(
            prompt="Describe the product in the image in less than 100 words.",
            batch=batch,
            max_new_tokens=256,
        )

        batch["color"] = self.generate(
            prompt="Return only the color of the product in the image in less than 10 words.",
            batch=batch,
            max_new_tokens=128,
        )

        return batch

    def generate(self, prompt, batch, max_new_tokens=128):

        messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": prompt}
            ]}
        ]

        input_text = self.processor.apply_chat_template(messages, add_generation_prmpt=True)
        outputs = []
        for item in batch["image"]:
            image = Image.fromarray(item)
            inputs = self.processor(
                image,
                input_text,
                add_special_tokens=False,
                return_tensors="pt"
            ).to(self.model.device)
            output = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
            output = self.processor.decode(output[0])
            outputs.append(output)

        return outputs

    

# COMMAND ----------

# MAGIC %md
# MAGIC ### Execute Inference
# MAGIC
# MAGIC Now that our class is ready, we can go ahead and begin the batch inference. 
# MAGIC
# MAGIC We will load the images in the same way, however as mentioned above, we will look to use the `map_batches` functionatility here with the loaded dataset. 

# COMMAND ----------

# Import Ray
import ray

# Init Ray
ray.init(ignore_reinit_error=True)

# Specify model path
model_path = "/Volumes/mas/item_onboarding/models/llama-32-11b-vision-instruct"

# Read images
image_analysis_df = ray.data.read_images(
    image_paths,
    include_paths=True,
    mode="RGB"
)

# Map batches
image_analysis_df = image_analysis_df.map_batches(
        LlamaVisionPredictor,
        concurrency=1,  # number of LLM instances
        num_gpus=1, # GPUs per LLM insatnce
        batch_size=10, # bach size 
        fn_constructor_kwargs={"model_path": model_path,},
)

# Evaluate
image_analysis_df = image_analysis_df.materialize()

# Determine where to save image analysis
save_path = "/Volumes/mas/item_onboarding/interim_data/image_analysis"

# Clear directory
dbutils.fs.rm(save_path, recurse=True)

# Save
image_analysis_df.write_parquet(save_path)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Display One Example
# MAGIC
# MAGIC Lets check out the results of our batch prediction too..

# COMMAND ----------

# Display Example
single_example = images.take(1)[0]

print(single_example["description"].split("<|start_header_id|>assistant<|end_header_id|>")[1].split("<|eot_id|>")[0].strip())
print(single_example["color"].split("<|start_header_id|>assistant<|end_header_id|>")[1].split("<|eot_id|>")[0].strip())

image = Image.fromarray(single_example["image"])
image

# COMMAND ----------

# MAGIC %md
# MAGIC ### Ray Shutdown
# MAGIC
# MAGIC We are done with the iamge analysis, and we can go ahead and shutdown RAY.

# COMMAND ----------

# Shutdown Ray
ray.shutdown()

# COMMAND ----------


