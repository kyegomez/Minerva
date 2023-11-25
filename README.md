# Minerva: Unleashing the Secrets of advanced Mathematics 🏛️🔢

<!-- ![Minerva Next Generation Open Source Language Model](/Minerva-banner.png) -->
Minerva is a groundbreaking language model that pushes the boundaries of mathematical understanding and problem-solving. Designed with an advanced math theme, Minerva embodies the spirit of renowned mathematicians such as Euclid, Pythagoras, and Archimedes. By harnessing their advanced wisdom, Minerva offers unparalleled capabilities in mathematical reasoning and exploration.

---

<div align="center">

[![GitHub issues](https://img.shields.io/github/issues/kyegomez/Minerva)](https://github.com/kyegomez/Minerva/issues) [![GitHub forks](https://img.shields.io/github/forks/kyegomez/Minerva)](https://github.com/kyegomez/Minerva/network) [![GitHub stars](https://img.shields.io/github/stars/kyegomez/Minerva)](https://github.com/kyegomez/Minerva/stargazers) [![GitHub license](https://img.shields.io/github/license/kyegomez/Minerva)](https://github.com/kyegomez/Minerva/blob/main/LICENSE)

</div>

<div align="center">

[![Share on Twitter](https://img.shields.io/twitter/url/https/twitter.com/cloudposse.svg?style=social&label=Share%20%40kyegomez/Minerva)](https://twitter.com/intent/tweet?text=Unleash%20the%20power%20of%20Minerva%20-%20the%20advanced-themed%20MATH%20LLM%20from%20Google!&url=https%3A%2F%2Fgithub.com%2Fkyegomez%2FMinerva) [![Share on Facebook](https://img.shields.io/badge/Share-%20facebook-blue)](https://www.facebook.com/sharer/sharer.php?u=https%3A%2F%2Fgithub.com%2Fkyegomez%2FMinerva) [![Share on LinkedIn](https://img.shields.io/badge/Share-%20linkedin-blue)](https://www.linkedin.com/shareArticle?mini=true&url=https%3A%2F%2Fgithub.com%2Fkyegomez%2FMinerva&title=&summary=&source=)

[![Share on Reddit](https://img.shields.io/badge/-Share%20on%20Reddit-orange)](https://www.reddit.com/submit?url=https%3A%2F%2Fgithub.com%2Fkyegomez%2FMinerva&title=Unleash%20the%20power%20of%20Minerva%20-%20the%20advanced-themed%20MATH%20LLM%20from%20Google!) [![Share on Hacker News](https://img.shields.io/badge/-Share%20on%20Hacker%20News-orange)](https://news.ycombinator.com/submitlink?u=https%3A%2F%2Fgithub.com%2Fkyegomez%2FMinerva&t=Unleash%20the%20power%20of%20Minerva%20-%20the%20advanced-themed%20MATH%20LLM%20from%20Google!) [![Share on Pinterest](https://img.shields.io/badge/-Share%20on%20Pinterest-red)](https://pinterest.com/pin/create/button/?url=https%3A%2F%2Fgithub.com%2Fkyegomez%2FMinerva&media=https%3A%2F%2Fexample.com%2Fimage.jpg&description=Unleash%20the%20power%20of%20Minerva%20-%20the%20advanced-themed%20MATH%20LLM%20from%20Google!) [![Share on WhatsApp](https://img.shields.io/badge/-Share%20on%20WhatsApp-green)](https://api.whatsapp.com/send?text=Unleash%20the%20power%20of%20Minerva%20-%20the%20advanced-themed%20MATH%20LLM%20from%20Google!%20%23Minerva%20%23AI%0A%0Ahttps%3A%2F%2Fgithub.com%2Fkyegomez%2FMinerva)

</div>

---





# Install

```shell
pip install minerva
```


# Usage
```python
import torch
from minerva import Minerva, Train

# Example usage
x = torch.randint(0, 20000, (1, 1024))

Minerva(x)

# or train
Train()
```

# Training

To train Minerva, follow these steps:

1. Configure the training settings by setting the environment variables:

   - `ENTITY_NAME`: Your wandb project name
   - `OUTPUT_DIR`: Specify the output directory for saving the weights (e.g., `./weights`)

2. Launch the training process using Deepspeed:

```shell
Accelerate Config
Accelerate launch train_distributed_accelerate.py
```

## Dataset Building

To build a custom dataset for Minerva, you can preprocess the data using the `build_dataset.py` script. This script performs tasks such as pre-tokenization, data chunking, and uploading to the Huggingface hub. Here's an example command:

| Dataset | Description |
|-|-|  
| Mathematical Web Pages | Web pages containing mathematical expressions in MathJax format, cleaned to preserve math notation|
| arXiv | 2 million arXiv papers up to Feb 2021, in LaTeX format |
| General Natural Language Data | Same dataset used to pretrain PaLM models |

The mathematical web pages and arXiv datasets focus on technical and mathematical content. The general natural language data provides a broad coverage of general language.

The paper states the mathematical web pages and arXiv each account for 47.5% of the total data. The remaining 5% is general natural language data which is a subset of what was used for PaLM pretraining.

## Roadmap 🗺️📍

- [ ] Create a dataset of ARXVIV papers