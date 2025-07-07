import logging


class CustomLogger():
    def __init__(self, name: str = "LLM-IDS", debug: bool = False):
        level = logging.DEBUG if debug is True else logging.INFO
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        
        # File handler
        file_handler = logging.FileHandler(f"results/{name}.log", mode='w', encoding='utf-8')
        file_handler.setFormatter(formatter)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def info(self, message: str):
        """
        Log an info message.
        :param message: The message to log.
        """
        self.logger.info(message)

    def debug(self, message: str):
        """
        Log a debug message.
        :param message: The message to log.
        """
        self.logger.debug(message)
    
    def error(self, message: str):
        """
        Log an error message.
        :param message: The message to log.
        """
        self.logger.error(message)