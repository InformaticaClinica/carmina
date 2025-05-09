from abc import ABC, abstractmethod
import os
from typing import Dict, Any, List, Optional
class BaseCloudProvider(ABC):
    def __init__(self, name: None, **kwargs):
        """
        Initializes the cloud provider with the necessary configurations.
        
        Args:
            **kwargs: Provider-specific parameters.
        """
        self.provider_name = name
    
    def get_name(self):
        """
        Returns the name of the cloud provider.
        Returns:
            str: Cloud provider name.
        """
        return self.provider_name

    @abstractmethod 
    def run_inference(self, model_id:str, messages:List[Dict[str, str]], **kwargs):
        """Runs inference on the deployed model"""
        pass