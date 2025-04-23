class DataLoadError(Exception):
    """Exception raised for errors in the data loading process.""" 
    def __init__(self, message="An error occurred while loading data"): 
        self.message = message
        super().__init__(message)