import os
import re
from groq import Groq  # Alterado de openai para groq
from typing import List
from utils.Logger import CustomLogger

class FeatureSelectorLLM():
    def __init__(self, logger: CustomLogger, model: str = "llama-3.1-8b-instant", temperature: float = 0.3):
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
        if not os.getenv("GROQ_API_KEY"):
            raise EnvironmentError("GROQ_API_KEY environment variable not set.")
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    def select_features(self, prompt: str) -> List[int]:
        """
        Use Groq LLM to select features based on a given prompt.
        :param prompt: A string describing the feature selection criteria.
        :return: A list of selected feature indices.
        """
        self.logger.info("Starting feature selection using Groq LLM...")

        try:
            # Query the LLM using the provided prompt
            llm_response = self._query_llm(prompt)

            # Parse the response to extract feature indices
            selected_features = self._parse_llm_response(llm_response)

            self.logger.info(f"Selected features: {selected_features}")
            return selected_features
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
            messages=[]
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
            return llm_response
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error querying Groq LLM: {e}")
            raise RuntimeError("Failed to query Groq LLM.") from e

    def _parse_llm_response(self, response: str) -> List[str]:
        """
        Parse the LLM response to extract only the list of selected features.
        :param response: The response from LLM as a string.
        :return: A list of selected feature names.
        """
        try:
            # Try to extract a Python list from a code block
            code_block = re.search(r"```python\s*selected_features\s*=\s*(\[[^\]]+\])", response, re.DOTALL)
            if code_block:
                list_str = code_block.group(1)
                # Extract feature names inside quotes
                features = re.findall(r'"([^"]+)"|\'([^\']+)\'', list_str)
                return [f1 or f2 for f1, f2 in features]

            # Fallback: extract a list enclosed in brackets
            bracket_list = re.search(r"\[([^\]]+)\]", response)
            if bracket_list:
                features = [f.strip(" '\"\n") for f in re.split(r',|\n', bracket_list.group(1)) if f.strip(" '\"\n")]
                return features

            raise ValueError("No feature names found in response.")
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error parsing LLM response: {e}")
            raise ValueError("Failed to parse LLM response.") from e
