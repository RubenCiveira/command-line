import os
import torch
import logging
from typing import List, Tuple, Optional, Dict
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from llama_cpp import Llama

class IntentClassifier:
	"""
	A class to handle intent classification using either the Llama.cpp model or a Hugging Face model.

	Parameters
	----------
	config : dict
		A configuration dictionary containing settings for model loading, storage, and model-specific options.
	"""

	def __init__(self, config: Dict):
		"""
		Initialize the IntentClassifier object.

		Parameters
		----------
		config : dict
			A configuration dictionary that includes model and storage settings.
		"""
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.config = config
		self.use_llama_cpp = config.get("use_llama_cpp", False)
		self.gcs = StorageHelper()
		self.logger = configure_logger()

		if self.use_llama_cpp:
			model_gcs_path = os.path.join(
				self.config["project_config"]["bucket"],
				self.config["gcs"]["repo_name"],
				self.config["gcs"]["model_folder"],
				self.config["gcs"]["model_path"],
				self.config["gcs"]["model_file"]
			)
			self._load_llama_cpp_model_from_gcs(model_gcs_path)
		else:
			model_gcs_path = os.path.join(
				self.config["project_config"]["bucket"],
				self.config["gcs"]["repo_name"],
				self.config["gcs"]["model_folder"],
				self.config["gcs"]["model_path"]
			)
			tokenizer_gcs_path = os.path.join(
				self.config["project_config"]["bucket"],
				self.config["gcs"]["repo_name"],
				self.config["gcs"]["model_folder"],
				self.config["gcs"]["tokenizer_path"]
			)
			self._load_model_and_tokenizer(model_gcs_path, tokenizer_gcs_path)

	def _load_llama_cpp_model_from_gcs(self, model_gcs_path: str):
		"""
		Download and load the llama.cpp model from GCS.

		Parameters
		----------
		model_gcs_path : str
			The Google Cloud Storage path to the llama.cpp model file.
		"""
		local_model_path = f"/tmp/{self.config['gcs']['model_folder']}/{self.config['gcs']['model_path']}"

		if os.path.exists(local_model_path):
			self.logger.info(f"Loading llama.cpp model from local path: {local_model_path}")
			self._load_llama_cpp_model(local_model_path)
		else:
			self.logger.info(f"Downloading llama.cpp model from GCS: {model_gcs_path}")
			self._download_single_file_from_gcs(model_gcs_path, local_model_path)
			self._load_llama_cpp_model(local_model_path)

	def _load_llama_cpp_model(self, model_path: str):
		"""
		Load the llama.cpp model from a local path using llama-cpp-python.

		Parameters
		----------
		model_path : str
			The path to the llama.cpp model file.
		"""
		try:
			self.logger.info(f"Loading llama.cpp model from {model_path}")
			self.llama_model = Llama(model_path, n_ctx=512)
			self.logger.info("Llama.cpp model loaded successfully")
		except Exception as e:
			self.logger.error(f"Error loading llama.cpp model: {e}")
			raise

	def _load_model_and_tokenizer(self, model_gcs_path: str, tokenizer_gcs_path: str):
		"""
		Load the Hugging Face model and tokenizer from GCS.

		Parameters
		----------
		model_gcs_path : str
			The GCS path to the model.
		tokenizer_gcs_path : str
			The GCS path to the tokenizer.
		"""
		local_model_path = f"/tmp/{self.config['gcs']['model_folder']}/model"
		local_tokenizer_path = f"/tmp/{self.config['gcs']['model_folder']}/tokenizer"

		if os.path.exists(local_model_path) and os.path.exists(local_tokenizer_path):
			self.logger.info("Loading model and tokenizer from local paths.")
			self._load_from_local(local_model_path, local_tokenizer_path)
		else:
			self.logger.info("Local paths not found, downloading from cloud storage.")
			self._download_from_gcs(model_gcs_path, local_model_path)
			self._download_from_gcs(tokenizer_gcs_path, local_tokenizer_path)
			self._load_from_local(local_model_path, local_tokenizer_path)

	def _download_single_file_from_gcs(self, gcs_path: str, local_path: str):
		"""
		Download a single file from GCS to a local path.

		Parameters
		----------
		gcs_path : str
			The GCS path to the file.
		local_path : str
			The local path where the file will be saved.
		"""
		try:
			bucket, directory = io.get_bucket_info(gcs_path)
			self.gcs.connect_to_bucket(bucket)

			if not os.path.exists(os.path.dirname(local_path)):
				os.makedirs(os.path.dirname(local_path))

			blob = self.gcs.bucket.blob(directory)
			blob.download_to_filename(local_path)
			self.logger.info(f"Downloaded {local_path} from GCS.")
		except Exception as e:
			self.logger.error(f"Error downloading from GCS: {e}")
			raise

	def _download_from_gcs(self, model_gcs_path: str, tokenizer_gcs_path: str, local_model_path: str, local_tokenizer_path: str):
		"""
		Download the model and tokenizer from GCS and save them locally.

		Parameters
		----------
		model_gcs_path : str
			The GCS path to the model files.
		tokenizer_gcs_path : str
			The GCS path to the tokenizer files.
		local_model_path : str
			The local directory where the model will be saved.
		local_tokenizer_path : str
			The local directory where the tokenizer will be saved.
		"""
		for gcs_path, local_path in [(model_gcs_path, local_model_path), (tokenizer_gcs_path, local_tokenizer_path)]:
			bucket, directory = io.get_bucket_info(gcs_path)
			self.gcs.connect_to_bucket(bucket)

			component = directory.split('/')[-1]
			self.logger.info(f"Component: {component}")

			if not os.path.exists(local_path):
				os.makedirs(local_path)

			blobs = list(self.gcs.bucket.list_blobs(prefix=directory))
			for blob in blobs:
				filename = blob.name.split('/')[-1]
				blob.download_to_filename(os.path.join(local_path, filename))
				self.logger.info(f"Downloaded {filename} from GCS.")
		
		self._load_from_local(local_model_path, local_tokenizer_path)


	def _load_from_local(self, model_path: str, tokenizer_path: str):
		"""
		Load the model and tokenizer from local paths.

		Parameters
		----------
		model_path : str
			The local path to the model.
		tokenizer_path : str
			The local path to the tokenizer.
		"""
		try:
			self.model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True).to(self.device)
			self.logger.info("Hugging Face model loaded successfully from local path!")
		except Exception as e:
			self.logger.error(f"Error loading Hugging Face model: {e}")
			raise

		try:
			self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
			self.logger.info("Tokenizer loaded successfully from local path.")
		except Exception as e:
			self.logger.error(f"Error loading tokenizer: {e}")
			try:
				self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
				self.logger.info("Tokenizer loaded from model path as fallback.")
			except Exception as fallback_e:
				self.logger.error(f"Error loading tokenizer from fallback: {fallback_e}")
				raise

		if self.tokenizer.pad_token is None:
			self.tokenizer.pad_token = self.tokenizer.eos_token

	def classify(self, query: str, prompt_template: str, few_shot_examples: Optional[List[Tuple[str, str]]] = None) -> str:
		"""
		Classify the intent of a given query using either llama.cpp or Hugging Face model.

		Parameters
		----------
		query : str
			The query to classify.
		prompt_template : str
			The template to use for generating prompts.
		few_shot_examples : Optional[List[Tuple[str, str]]], optional
			A list of few-shot examples to include in the prompt, by default None.

		Returns
		-------
		str
			The predicted intent for the query.
		"""
		try:
			if few_shot_examples:
				examples_text = "\n".join(f"Query: {example}\nIntent: {label}" for example, label in few_shot_examples)
				prompt = prompt_template.format(examples=examples_text, query=query)
			else:
				prompt = prompt_template.format(query=query)

			if self.use_llama_cpp:
				return self._llama_cpp_classify(prompt)
			else:
				return self._hf_classify(prompt)

		except Exception as e:
			self.logger.error(f"Error classifying query: {e}")
			raise

	def _llama_cpp_classify(self, prompt: str) -> str:
		"""
		Classify a query using the llama.cpp model.

		Parameters
		----------
		prompt : str
			The formatted prompt for the llama.cpp model.

		Returns
		-------
		str
			The predicted intent from the model output.
		"""
		try:
			output = self.llama_model(prompt, max_tokens=10, temperature=0.0)
			generated_text = output["choices"][0]["text"].strip()
			self.logger.info(f"Generated text: {generated_text}")

			generated_lines = generated_text.strip().split("\n")
			for line in generated_lines:
				line = line.strip()
				if line:
					predicted_intent = line.strip()
					break

			return predicted_intent
		except Exception as e:
			self.logger.error(f"Error in llama.cpp classification: {e}")
			raise

	def _hf_classify(self, prompt: str) -> str:
		"""
		Classify a query using the Hugging Face model.

		Parameters
		----------
		prompt : str
			The formatted prompt for the Hugging Face model.

		Returns
		-------
		str
			The predicted intent from the model output.
		"""
		inputs = self.tokenizer.encode_plus(
			prompt,
			return_tensors='pt',
			padding=True,
			truncation=True,
			max_length=512,
		)

		input_ids = inputs['input_ids'].to(self.device)
		attention_mask = inputs['attention_mask'].to(self.device)

		with torch.no_grad():
			outputs = self.model.generate(
				input_ids,
				attention_mask=attention_mask,
				max_new_tokens=10,
				num_return_sequences=1,
				temperature=0.3,
				eos_token_id=self.tokenizer.encode('\n')
			)

		generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
		predicted_intent = generated_text.split("Intent:")[-1].strip().split()[0]

		return predicted_intent

	def batch_classify(self, queries: List[str], prompt_template: str, few_shot_examples: Optional[List[Tuple[str, str]]] = None) -> List[str]:
		"""
		Batch classify a list of queries.

		Parameters
		----------
		queries : List[str]
			A list of queries to classify.
		prompt_template : str
			The template to use for generating prompts.
		few_shot_examples : Optional[List[Tuple[str, str]]], optional
			A list of few-shot examples to include in the prompts, by default None.

		Returns
		-------
		List[str]
			A list of predicted intents for each query.
		"""
		try:
			results = []
			for query in queries:
				intent = self.classify(query, prompt_template, few_shot_examples)
				results.append(intent)
			return results
		except Exception as e:
			self.logger.error(f"Error in batch classification: {e}")
			raise
