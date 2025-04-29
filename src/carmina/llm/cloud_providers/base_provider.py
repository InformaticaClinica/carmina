class BaseCloudProvider:
    def __init__(self, **kwargs):
        """
        Initializes the cloud provider with the necessary configurations.
        
        Args:
            **kwargs: Provider-specific parameters.
        """
        self.provider_name = kwargs.get("provider_name", "base") # Default name
    
    def get_name(self):
        """
        Returns the name of the cloud provider.
        Returns:
            str: Cloud provider name.
        """
        return self.provider_names
        
    def run_inference(self, model_id, messages, **kwargs):
        """Runs inference on the deployed model"""
        pass