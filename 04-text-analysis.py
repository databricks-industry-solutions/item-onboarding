# Databricks notebook source
# MAGIC %md
# MAGIC # Text Analysis
# MAGIC
# MAGIC We have completed the image analysis in the previous section which gave us some more data points such as the observed description and observed color. Now, we can use a text based LLM model to tidy all the text up.
# MAGIC
# MAGIC The goal of this notebook is to create the resulting text points such as the final description or the final color by considering all of the information that has been gathered, either by the supplier or through our workflow.
# MAGIC
# MAGIC As far as the code goes, we are going to follow a very similar flow with the exception of [vLLM](https://docs.vllm.ai/en/latest/). VLLM is a very popular library that helps with optimizing the models during runtime. It works with almost all of the SOTA Open Source models. It actually has an experimental application for our vision model too, however that is not just yet ready for production therefore we didn't use it in our previous notebook.
# MAGIC
# MAGIC The way to design the code is slightly different with vLLM, especially at the point where we need to call the model, however the rest that rotates around Ray is pretty much the same.
# MAGIC
# MAGIC We are going ot follow a similar flow here where we are going to do some interactive testing with the prompts to begin with, and then we are going ot design the necessary flow for the batch inference.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Library Install
# MAGIC
# MAGIC Installing the necessary libraries here, transformers and vllm

# COMMAND ----------

# MAGIC %sh
# MAGIC # Installing necessary libraries for model & inference
# MAGIC pip install --upgrade transformers -q
# MAGIC pip install vllm -q

# COMMAND ----------

# This operation has to be in a seperate cell than library installation cell above
dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Set the Defaults
# MAGIC
# MAGIC Specify the default Unity Catalog and the Schema, as well as create a Volume to store the interim data and the path which will hold the onboarding df.

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Defining the defaults
# MAGIC USE CATALOG mas;
# MAGIC USE SCHEMA item_onboarding;
# MAGIC -- Creating storage location for interim data
# MAGIC CREATE VOLUME IF NOT EXISTS interim_data;

# COMMAND ----------

# Specify target path
onboarding_df_path = "/Volumes/mas/item_onboarding/interim_data/onboarding"

# COMMAND ----------

# MAGIC %md
# MAGIC ### Build Interim Data
# MAGIC
# MAGIC We need to take in all of the data points, the ones we got from the suppliers as well as the ones we got from the visual model and then need to join them so we can use it all during the text based workflow.
# MAGIC
# MAGIC It is easier for Ray to pick up Parquet files from Databricks' Volumes, therefore, at the very end of the cell, we will save the finalised interim dataframe as a Parquet on the Volume.

# COMMAND ----------

from pyspark.sql import functions as SF

# Build to be processed table in parquet
products_clean_df = spark.read.table("mas.item_onboarding.products_clean_sampled")
image_meta_df = spark.read.table("mas.item_onboarding.image_meta_enriched_sampled")
image_analysis_df = spark.read.parquet("/Volumes/mas/item_onboarding/interim_data/image_analysis")

# Basic Transformations
image_analysis_df = (
    image_analysis_df
    .drop("image")
    .selectExpr([
        "path AS real_path", 
        "description AS gen_description", 
        "color AS gen_color",
    ]
)

# Cleaning the generated description and color text
pattern = r"assistant<\|end_header_id\|>\s*([\s\S]*?)<\|eot_id\|>"
image_analysis_df = (
    image_analysis_df
    .withColumn("gen_description", SF.regexp_extract("gen_description", pattern, 1))
    .withColumn("gen_color", SF.regexp_extract("gen_color", pattern, 1))
)

# Prepare entry for image desc mixing for later
onboarding_df = (
    products_clean_df
    .join(image_meta_df, on="item_id", how="left")
    .join(image_analysis_df, on="real_path", how="left")
)

# Save as parquet at the created location
(
    onboarding_df
    .write
    .mode("overwrite")
    .parquet(onboarding_df_path)
)
# display(onboarding_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Build Target Product Taxonomy
# MAGIC
# MAGIC In this section, we also wanted to simulate a scenario where the Retailer might have a pre-defined taxonomy for their catalog. The task down the line will be to place the items within this taxonomy. This usually helps Retailers categorise their products. So, we thought that we would generate a real life like taxonomy and see how our model could perform with it.

# COMMAND ----------

product_taxonomy = """- Furniture & Home Furnishings - Chairs
- Furniture & Home Furnishings - Tables
- Furniture & Home Furnishings - Sofas & Couches
- Furniture & Home Furnishings - Cabinets, Dressers & Wardrobes
- Furniture & Home Furnishings - Lamps & Light Fixtures
- Furniture & Home Furnishings - Shelves & Bookcases
- Footwear & Apparel - Shoes
- Footwear & Apparel - Clothing
- Footwear & Apparel - Accessories
- Kitchen & Dining - Cookware
- Kitchen & Dining - Tableware
- Kitchen & Dining - Cutlery & Utensils
- Kitchen & Dining - Storage & Organization
- Home Décor & Accessories - Vases & Decorative Bowls
- Home Décor & Accessories - Picture Frames & Wall Art
- Home Décor & Accessories - Decorative Pillows & Throws
- Home Décor & Accessories - Rugs & Mats
- Consumer Electronics - Headphones & Earbuds
- Consumer Electronics - Portable Speakers
- Consumer Electronics - Keyboards, Mice & Other Peripherals
- Consumer Electronics - Phone Cases & Stands
- Office & Stationery - Desk Organizers & Pen Holders
- Office & Stationery - Notebooks & Journals
- Office & Stationery - Pens, Pencils & Markers
- Office & Stationery - Folders, Binders & File Organizers
- Personal Care & Accessories - Water Bottles & Tumblers
- Personal Care & Accessories - Makeup Brushes & Hair Accessories
- Personal Care & Accessories - Personal Grooming Tools
- Toys & Leisure - Action Figures & Dolls
- Toys & Leisure - Building Blocks & Construction Sets
- Toys & Leisure - Board Games & Puzzles
- Toys & Leisure - Plush & Stuffed Animals"""

# COMMAND ----------

# MAGIC %md
# MAGIC ### Interactive Model & Prompt Configuration
# MAGIC
# MAGIC We will now begin the interactive part with our text model. Our goal here is to test how the model works, as well as do some prompt engineering for the text analysis.
# MAGIC
# MAGIC Similar to the way we did with the image model, we will go ahead and create an actor. The difference here is that we will use the vLLM library to load our model. Since vLLM is optimised, we can expect faster model loading (from Volume to the GPU memory) as well as faster inference.

# COMMAND ----------

# Imports
from vllm import LLM, SamplingParams
import ray

# Init Ray
ray.init(ignore_reinit_error=True)

# Specify model path
model_path = "/Volumes/mas/item_onboarding/models/llama-31-8b-instruct/"


# Load the LLM to the GPU
@ray.remote(num_gpus=1)
class LLMActor:
    def __init__(self, model_path: str):
        # Initiate the model
        self.model = LLM(model=model_path, max_model_len=2048)

    def generate(self, prompt, sampling_params):
        raw_output = self.model.generate(prompt, sampling_params=sampling_params)
        return raw_output


# Create the LLM actor - this part loads the model to the GPU. It will do it async.
llm_actor = LLMActor.remote(model_path)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Prompting Techniques
# MAGIC
# MAGIC We are using the LLAMA 3.1 8B instruct model. This model expects to be called in a specific way which is a little different than the base model. This special way requires for us to format our prompt or our instruction with special tokens and a preset strucutre. The structure expects to receive a system prompt, which tells the model somethig like "you are a helpful assistant". The same goes for the instruction. Both of the text pieces are then placed before/after special tokens, which look something like: `<|eot_id|>`. For more information on this technique, we can check out [Meta's Model docs](https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/prompt_format.md)
# MAGIC
# MAGIC In the cell below, we create a basic function which can build the prompt in the right format given the system and the instruction text.

# COMMAND ----------


# Llama prompt format
def produce_prompt(system, instruction):
    prompt = (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
        f"{system}<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
        f"{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
    )
    return prompt


test_prompt = produce_prompt(
    "You are a helpful assistant", "How many days are there in a week"
)
print(test_prompt)

# COMMAND ----------

# MAGIC %md
# MAGIC Following this, lets do a simple test:

# COMMAND ----------

# Calling the actor with the generation request built above
result = ray.get(
    llm_actor.generate.remote(test_prompt, SamplingParams(temperature=0.1))
)

# Formatting result printing
print(result)

# The actual result object is a list of outputs, so we need to access the first one
print("\n")
print(result[0].prompt)
print("\n")
print(" ".join([o.text for o in result[0].outputs]).strip())
print("\n")

# COMMAND ----------

# MAGIC %md
# MAGIC Our model is working, let start testing on some real examples. Loading our actual dataset in the cell below.

# COMMAND ----------

# Read the dataset to get some examples
onboarding_ds = ray.data.read_parquet(onboarding_df_path)

# # Show its schema
print(onboarding_ds.schema())

# COMMAND ----------

# MAGIC %md
# MAGIC Lets check out what a single record looks like from this dataset

# COMMAND ----------

# Get a single record
single_record = onboarding_ds.take(2)[1]
print(single_record)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Generic Sampling Params
# MAGIC
# MAGIC Sampling params can be used adjust the output of the model. There are many arguments that can adjusted for here. Depending on the configuration, for example the temprature, we can make the model more "creative" or more "instuction following". We can adjust the token selection process by changing the top_p and top_k parameters, or decide how long of and answer the model can return to us by changing the max_tokens. More information on this can be found on the [vLLM Sampling Params](https://docs.vllm.ai/en/stable/dev/sampling_params.html).

# COMMAND ----------

sampling_params = SamplingParams(
    n=1,  # Number of output sequences to return for the given prompt
    temperature=0.1,  # Randomness of the sampling. Lower values make the model more deterministic, while higher values make the model more random. Zero means greedy sampling.
    top_p=0.9,  # Cumulative probability of the top tokens to consider
    top_k=50,  # Number of top tokens to consider
    max_tokens=256,  # Adjust this value based on your specific task
    stop_token_ids=[128009],  # Stop the generation when they are generated
    presence_penalty=0.1,  # Penalizes new tokens based on whether they appear in the generated text so far
    frequency_penalty=0.1,  # Penalizes new tokens based on their frequency in the generated text so far
    ignore_eos=False,  # Whether to ignore the EOS token and continue generating tokens after the EOS token is generated.
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Description Prompt
# MAGIC
# MAGIC Lets begin with our description prompt. Given the visual description generated by the image model, and the information received from the suplier, we will ask the model to generate a new description.

# COMMAND ----------

# Suggested description - system prompt
description_system_prompt = "You are an expert retail product writer."

# Suggested description - instruction prompt
description_instruction = """
Below are two descriptions for a product. Create a natural and clear description (<50 words) that captures the key details.

Description 1: {bullet_point}
Description 2: {gen_description}

Output only the new description. No quotes or additional text.
"""

# Populate the prompt
description_instruction = description_instruction.format(
    bullet_point=single_record["bullet_point"],
    gen_description=single_record["gen_description"],
)

# Format the prompt
description_prompt = produce_prompt(
    system=description_system_prompt, instruction=description_instruction
)

print(description_prompt)

result = ray.get(llm_actor.generate.remote(description_prompt, sampling_params))
suggested_description = " ".join([o.text for o in result[0].outputs]).strip()
print(suggested_description)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Color Prompt
# MAGIC
# MAGIC Our description is ready, lets go ahead and ask the model to generate an ultimate color for our product. Some of the data coming from the supplier is missing the color field, so the input from the visual model is going to be key here.

# COMMAND ----------

# Suggested color - system prompt
color_system_prompt = "You are an expert color analyst."

# Suggested color - instruction prompt
color_instruction = """
Given:
- Product color: {color}
- Vision model color: {gen_color}

Return the color. No extra text.
"""


# Populate the prompt
color_instruction = color_instruction.format(
    color=single_record["color"],
    gen_color=single_record["gen_color"],
)

# Format the prompt
color_prompt = produce_prompt(system=color_system_prompt, instruction=color_instruction)

print(color_prompt)

result = ray.get(llm_actor.generate.remote(color_prompt, sampling_params))
suggested_color = " ".join([o.text for o in result[0].outputs]).strip()
print(suggested_color)

# COMMAND ----------

# MAGIC %md
# MAGIC This is a great example because the color provided by the supplier, "hunter", is not an actual color. The vision model confirms that the actual color is "green" and thats what the text model actually decides.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Keyword Prompt
# MAGIC
# MAGIC Our suppliers also give us bunch of keywords to optimise for search, however there are problematic data points coming from here too where keywords get repeated multiple times, or don't actually match the item correctly.
# MAGIC
# MAGIC This part will aim to optimize the keywords while keeping the same format.

# COMMAND ----------

# Suggested keyword - system prompt
keyword_system_prompt = "You are an expert SEO and product keyword specialist."

# Suggested keyword - instruction prompt
keyword_instruction = """
Input:
- Current keywords: {item_keywords}
- Product description: {suggested_description}
- Product color: {suggested_color}

Return new keywords separated by |. No other text. Do not explain.
"""


# Format the prompt
keyword_prompt = produce_prompt(
    system=keyword_system_prompt, instruction=keyword_instruction
)

# Populate the prompt
keyword_prompt = keyword_prompt.format(
    item_keywords=single_record["item_keywords"],
    suggested_description=suggested_description,
    suggested_color=suggested_color,
)


print(keyword_prompt)

result = ray.get(llm_actor.generate.remote(keyword_prompt, sampling_params))
suggested_keywords = " ".join([o.text for o in result[0].outputs]).strip()
print("\n")
print(suggested_keywords)

# COMMAND ----------

# MAGIC %md
# MAGIC and, the model is successfully able to do that too!

# COMMAND ----------

# MAGIC %md
# MAGIC ### Category Prompt
# MAGIC
# MAGIC Finally, after generating and correcting all this information, we are going to model to put the item in one of the categories we have created at the top of the notebook.
# MAGIC
# MAGIC In this part, the model will use the text it has generated for us from the previous cells too.

# COMMAND ----------

# Suggested taxonomy - system prompt
taxonomy_system_prompt = "You are an expert merchandise taxonomy specialists"

# Suggested taxonomy - instruction prompt
taxonomy_instruction = """
Review the product description and choose the most suitable category from the provided taxonomy.
Product Description: 
{suggested_description}

Product Taxonomy: 
{target_taxonomy}

Return the single best matching category and no other text.
"""

# Format the prompt
taxonomy_prompt = produce_prompt(
    system=taxonomy_system_prompt, instruction=taxonomy_instruction
)

# Populate the prompt
taxonomy_prompt = taxonomy_prompt.format(
    suggested_description=suggested_description,
    target_taxonomy=product_taxonomy,
)

print(taxonomy_prompt)

result = ray.get(llm_actor.generate.remote(taxonomy_prompt, sampling_params))
suggested_category = " ".join([o.text for o in result[0].outputs]).strip()

print("\n")
print(suggested_category)


# COMMAND ----------

# MAGIC %md
# MAGIC The model succesfully places the item in the right category.

# COMMAND ----------

# MAGIC %md
# MAGIC ### GPU Unload
# MAGIC
# MAGIC We are ready for batch inference, and will unload the GPU by shutting down Ray before we continue

# COMMAND ----------

ray.shutdown()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Batch Inference
# MAGIC
# MAGIC Now that we have interactively worked with our model and understood how our prompts can work with the model, it is time to set the flow up for batch inference.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Ray Init & Data Pick Up
# MAGIC
# MAGIC We will go ahead and re-init ray and pick up the dataset for batch inference.

# COMMAND ----------

# Imports
import ray

# Init ray
ray.init()

# Pick up the data
onboarding_ds = ray.data.read_parquet(onboarding_df_path)

# Inspect Schema
onboarding_ds.schema

# COMMAND ----------

# MAGIC %md
# MAGIC ### Inference Logic
# MAGIC
# MAGIC The way we will design our inference is going to be quite similar to the way we have done so with the image model, with the exception ,again, being the fact that we will use vLLM here..
# MAGIC
# MAGIC We will use the class with `__init__` and `__call__` methods, where the `__call__` method will hold the flow of our inference. The flow is important as the answers generated in the first steps will be used in the later stages, so it needs to be sequential.
# MAGIC
# MAGIC We will also build some helper functions to standardise things like prompt formatting.

# COMMAND ----------

# Imports
from vllm import LLM, SamplingParams
import numpy as np


class OnboardingLLM:
    # Building the class here
    def __init__(self, model_path: str, target_taxonomy: str):
        # Initiate the model
        self.model = LLM(model=model_path, max_model_len=2048)
        self.target_taxonomy = target_taxonomy

    def __call__(self, batch):
        """Define the logic to be executed on each batch"""
        # All inference logic will go here
        batch = self.generate_suggested_description(batch)
        batch = self.generate_suggested_color(batch)
        batch = self.generate_suggested_keywords(batch)
        batch = self.generate_suggested_product_category(batch)
        return batch

    @staticmethod
    def format_prompt(system, instruction):
        """Helps with formatting the prompts"""
        prompt = (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
            f"{system}<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
            f"{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
        )
        return prompt

    @staticmethod
    def standardise_output(raw_output):
        """Return standardised output after each inference"""
        generated_outputs = []
        for _ro in raw_output:
            generated_outputs.append(" ".join([o.text for o in _ro.outputs]))
        return generated_outputs

    @staticmethod
    def build_sampling_params(max_tokens=256):
        """Build sampling params for inference"""
        sampling_params = SamplingParams(
            n=1,
            temperature=0.1,
            top_p=0.9,
            top_k=50,
            max_tokens=max_tokens,  # Adjust this value based on your specific task
            stop_token_ids=[128009],  # Specific to LLAMA 3.1 <|eot_id|>
            presence_penalty=0.1,
            frequency_penalty=0.1,
            ignore_eos=False,
        )
        return sampling_params

    def generate_suggested_description(self, batch):
        # Suggested description - system prompt
        system_prompt = "You are an expert retail product writer."

        # Suggested description - instruction prompt
        instruction = """
        Below are two descriptions for a product. Create a natural and clear description (<50 words) that captures the key details.

        Description 1: {bullet_point}
        Description 2: {gen_description}

        Output only the new description. No quotes or additional text.
        """

        # Build prompts
        prompt_template = self.format_prompt(
            system=system_prompt, instruction=instruction
        )
        prompts = np.vectorize(prompt_template.format)(
            bullet_point=batch["bullet_point"], gen_description=batch["gen_description"]
        )

        # Build sampling params
        sampling_params = self.build_sampling_params(max_tokens=256)

        # Inference
        raw_output = self.model.generate(prompts, sampling_params=sampling_params)

        # Return to batch
        batch["suggested_description"] = self.standardise_output(raw_output)

        return batch

    def generate_suggested_color(self, batch):
        # Suggested color - system prompt
        system_prompt = "You are an expert color analyst."

        # Suggested color - instruction prompt
        instruction = """
        Given a product's :
        - Described color: {color}
        - Observed color: {gen_color}

        Return the color. No extra text.
        """

        # Format the prompt
        prompt_template = self.format_prompt(
            system=system_prompt, instruction=instruction
        )
        prompts = np.vectorize(prompt_template.format)(
            color=batch["color"], gen_color=batch["gen_color"]
        )

        # Build sampling params
        sampling_params = self.build_sampling_params(max_tokens=16)

        # Inference
        raw_output = self.model.generate(prompts, sampling_params=sampling_params)

        # Return to batch
        batch["suggested_color"] = self.standardise_output(raw_output)

        return batch

    def generate_suggested_keywords(self, batch):
        # Suggested keyword - system prompt
        system_prompt = "You are an expert SEO and product keyword specialist."

        # Suggested keyword - instruction prompt
        instruction = """
        Input:
        - Current keywords: {item_keywords}
        - Product description: {suggested_description}
        - Product color: {suggested_color}

        Return new keywords separated by |. No other text. Do not explain.
        """

        # Format the prompt
        prompt_template = self.format_prompt(
            system=system_prompt, instruction=instruction
        )
        prompts = np.vectorize(prompt_template.format)(
            item_keywords=batch["item_keywords"],
            suggested_description=batch["suggested_description"],
            suggested_color=batch["suggested_color"],
        )

        # Build sampling params
        sampling_params = self.build_sampling_params(max_tokens=256)

        # Inference
        raw_output = self.model.generate(prompts, sampling_params=sampling_params)

        # Return to batch
        batch["suggested_keywords"] = self.standardise_output(raw_output)

        return batch

    def generate_suggested_product_category(self, batch):
        # Suggested category - system prompt
        system_prompt = "You are an expert merchandise taxonomy specialists"

        # Suggested category - instruction prompt
        instruction = """
        Review the product description and choose the most suitable category from the provided taxonomy.
        Product Description: 
        {suggested_description}

        Product Taxonomy: 
        {target_taxonomy}

        Return the single best matching category and no other text.
        """

        # Format the prompt
        prompt_template = self.format_prompt(
            system=system_prompt, instruction=instruction
        )
        prompts = np.vectorize(prompt_template.format)(
            suggested_description=batch["suggested_description"],
            target_taxonomy=self.target_taxonomy,
        )

        # Build sampling params
        sampling_params = self.build_sampling_params(max_tokens=256)

        # Inference
        raw_output = self.model.generate(prompts, sampling_params=sampling_params)

        # Return to batch
        batch["suggested_category"] = self.standardise_output(raw_output)

        return batch


# COMMAND ----------

# MAGIC %md
# MAGIC ### Execute Inference
# MAGIC
# MAGIC Our class is ready, we can fire up the inference logic!
# MAGIC
# MAGIC We will again save the results as Parquet files on the Volumes, and will pick these up in the next notebook to view the results.

# COMMAND ----------

# Specify model path
model_path = "/Volumes/mas/item_onboarding/models/llama-31-8b-instruct/"

# Pick up the data
onboarding_ds = ray.data.read_parquet(onboarding_df_path)

# Run the flow
ft_onboarding_ds = onboarding_ds.map_batches(
    OnboardingLLM,
    concurrency=1,  # number of LLM instances
    num_gpus=1,  # GPUs per LLM instance
    batch_size=32,  # maximize until OOM, if OOM then decrease batch_size
    fn_constructor_kwargs={
        "model_path": model_path,
        "target_taxonomy": product_taxonomy,
    },
)

# Evaluate
ft_onboarding_ds = ft_onboarding_ds.materialize()

# Determine where to save results
save_path = "/Volumes/mas/item_onboarding/interim_data/results"

# Clear the folder
dbutils.fs.rm(save_path, recurse=True)

# Save
ft_onboarding_ds.write_parquet(save_path)

# COMMAND ----------

ray.shutdown()

# COMMAND ----------
