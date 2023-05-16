class BaseAgent:
    """A wrapper module made according the api of stable-baselines3."""
    def learn(self, *args, **kwargs):
        self.model.learn(*args, **kwargs)
        
    def save(self, *args, **kwargs):
        self.model.save(*args, **kwargs)
        
    def load(self, *args, **kwargs):
        return self.model.__class__.load(*args, **kwargs)