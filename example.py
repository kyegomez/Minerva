import torch 
from minerva.model import Minerva

# Usage with random inputs
text = torch.randint(0, 20000, (1, 1024))

# Initiliaze the model
model = Minerva()
output = model(text)
print(output)
