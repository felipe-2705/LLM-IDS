import os
import re
import openai
from typing import List

class FeatureSelectorLLM():
    def __init__(self, logger=None, model: str = "gpt-3.5-turbo", temperature: float = 0.3):
        """
        Initialize the FeatureSelectorLLM with a model and optional logger.
        :param logger: A CustomLogger object for logging operations (optional).
        :param model: Name of the LLM model to use (default is gpt-3.5-turbo).
        :param temperature: Sampling temperature for the LLM (default is 0.3).
        """
        self.logger = logger
        self.model = model
        self.temperature = temperature
        self.history = []

    def select_features(self, prompt: str) -> List[int]:
        """
        Use ChatGPT to select features based on a given prompt.
        :param prompt: A string describing the feature selection criteria.
        :return: A list of selected feature indices.
        """
        if self.logger:
            self.logger.info("Starting feature selection using ChatGPT...")

        try:
            # Query the LLM using the provided prompt
            llm_response = self._query_llm(prompt)

            # Parse the response to extract feature indices
            selected_features = self._parse_llm_response(llm_response)

            if self.logger:
                self.logger.info(f"Selected features: {selected_features}")

            return selected_features
        except Exception as e:
            if self.logger:
                self.logger.error(f"Feature selection failed: {e}")
            raise RuntimeError("Feature selection process failed.") from e

    def _query_llm(self, prompt: str) -> str:
        """
        Query OpenAI's ChatGPT API to get a response based on the prompt.
        :param prompt: A string describing the feature selection criteria.
        :return: The response from ChatGPT as a string.
        """
        try:
            # Load API key from environment variable
            openai.api_key = os.getenv("OPENAI_API_KEY")
            if not openai.api_key:
                raise EnvironmentError("OPENAI_API_KEY environment variable not set.")
            
            messages = self.history.copy()
            messages.append({"role": "user", "content": prompt})
            response = openai.ChatCompletion.create(
                model=self.model,
                temperature=self.temperature,
                messages=messages
            )
            llm_response = response['choices'][0]['message']['content']
            self.history.append({"role": "user", "content": prompt})
            self.history.append({"role": "assistant", "content": llm_response})
            return llm_response
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error querying ChatGPT: {e}")
            raise RuntimeError("Failed to query ChatGPT.") from e

    def _parse_llm_response(self, response: str) -> List[int]:
        """
        Parse the ChatGPT response to extract feature indices.
        :param response: The response from ChatGPT as a string.
        :return: A list of feature indices extracted from the response.
        """
        try:
            # Use regex to find all integers in the response
            matches = re.findall(r'\b\d+\b', response)
            indices = [int(m) for m in matches]
            if not indices:
                raise ValueError("No feature indices found in response.")
            return indices
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error parsing ChatGPT response: {e}")
            raise ValueError("Failed to parse ChatGPT response.") from e
