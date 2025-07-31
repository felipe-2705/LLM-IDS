import os
import re
from groq import Groq  # Alterado de openai para groq
from typing import List
import json
from utils.Logger import CustomLogger

class FeatureSelectorLLM():
    def __init__(self, logger: CustomLogger, model: str = "llama-3.1-8b-instant", temperature: float = 0.7):
        """
        Initialize the FeatureSelectorLLM with a model and optional logger.
        :param logger: A CustomLogger object for logging operations (optional).
        :param model: Name of the Groq model to use (default is llama-3.1-8b-instant).
        :param temperature: Sampling temperature for the LLM (default is 0.3).
        """
        self.logger = logger
        self.model = model
        self.temperature = temperature
        self.history = []
        self.key_counter = 0
        self.keys = self.read_keys("groq-key.json")  # Read keys from the JSON file
        self.client = Groq(api_key=self.get_next_key())  # Initialize Groq client with the first key

    def read_keys(self, file_path: str):
        """
        Read Groq API keys from a JSON file.
        :param file_path: Path to the JSON file containing the keys.
        :return: A list of API keys.
        """
        try:
            with open(file_path, 'r') as file:
                data = json.load(file)
                return data.get("keys", [])
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error reading Groq keys from {file_path}: {e}")
            raise RuntimeError("Failed to read Groq keys.") from e

    def get_next_key(self):
        """
        Get a Groq API key from the list of keys.
        :return: A Groq API key.
        """
        if not self.keys:
            raise ValueError("No Groq API keys available.")
        
        key = self.keys[1]
        self.key_counter += 1
        return key

    def select_features(self, prompt: str,valid_features) -> List[int]:
        """
        Use Groq LLM to select features based on a given prompt.
        :param prompt: A string describing the feature selection criteria.
        :return: A list of selected feature indices.
        """
        self.logger.info("Starting feature selection using Groq LLM...")
        invalid_solutions = 0
        try:
            # Query the LLM using the provided prompt
            valid_set = False
            while not valid_set:
                self.logger.debug(f"Querying LLM with prompt: {prompt}")
                llm_response = self._query_llm(prompt)
                # Parse the response to extract feature indices

                self.logger.debug(f"LLM Response: {llm_response}")

                solutions = self._parse_llm_response(llm_response)
                if solutions is None:
                    self.logger.warning("LLM response did not contain valid JSON structure. Retrying...")
                    invalid_solutions += 1
                    continue
                valid_set = True
                for solution in solutions:
                    for feature in solution:
                        if feature not in valid_features:
                            self.logger.warning(f"Feature '{feature}' not in valid features, retrying...")
                            valid_set = False
                            prompt+=f" Feature '{feature}' not in valid features"
                            invalid_solutions+=1
                            break
            self.logger.info(f"Selected features: {solutions}")
            return solutions,invalid_solutions
        except Exception as e:
            self.logger.error(f"Feature selection failed: {e}")
            raise RuntimeError("Feature selection process failed.") from e

    def _query_llm(self, prompt: str) -> str:
        """
        Query Groq's LLM API to get a response based on the prompt.
        :param prompt: A string describing the feature selection criteria.
        :return: The response from Groq LLM as a string.
        """
        try:
            messages=[{
                        "role": "system",
                        "content": (
                            "You are a strict assistant that only replies in the exact format requested. "
                            "Never add extra explanation, code formatting, or comments. "
                            "Only return valid JSON if asked, and always respect numerical limits (e.g., MAX 10 items). "
                            "If the user requests something structured, follow the structure precisely."
                        )
                        }]
            #messages = self.history.copy()
            messages.append({"role": "user", "content": prompt})
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                messages=messages,
                stream=False
            )
            llm_response = response.choices[0].message.content
            #self.history.append({"role": "user", "content": prompt})
            #self.history.append({"role": "assistant", "content": llm_response})
            self.client.api_key = self.get_next_key()  # Set the API next key for the client
            return llm_response
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error querying Groq LLM: {e}")
            raise RuntimeError("Failed to query Groq LLM.") from e

    def _parse_llm_response(self, response: str) -> List[List[str]]:
        """
        Parse the LLM response to extract a list of solutions, where each solution is a list of feature names.
        Expected format: '{"solutions": [["feat1", "feat2"], ["feat3", "feat4"]]}'.
        
        :param response: The response from the LLM as a string.
        :return: A list of solutions, each being a list of feature names.
        """
        try:
            # Try to extract JSON block from response
            json_match = re.search(r'\{.*"solutions"\s*:\s*\[.*\]\s*\}', response, re.DOTALL)
            
            if json_match:
                json_str = json_match.group(0)
                try:
                    data = json.loads(json_str)
                except json.JSONDecodeError:
                    return None # Return None if JSON parsing fails
                if "solutions" in data and isinstance(data["solutions"], list):
                    if all(isinstance(sol, list) for sol in data["solutions"]):
                        return data["solutions"]

            return None  # Return None if no valid JSON structure is found
        except Exception as e:
            return None  # Return None if parsing fails
