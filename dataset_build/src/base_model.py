class BaseModel:
    def __init__(self, name, path, parameters):
        self.name = name
        self.parameters = parameters
        self.model = self.load_model(path, name)

    def load_model(self, path, name):
        raise NotImplementedError("Subclasses must implement load_model method")

    def generate(self, instruction):
        raise NotImplementedError("Subclasses must implement generate method")
