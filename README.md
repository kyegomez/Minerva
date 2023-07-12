# [READY FOR TRAINING, help us with the strategy!](https://www.figma.com/file/pfaU8Nhyw0EdXuT6z4Hutw/Minerva-Strategy?type=whiteboard&node-id=0%3A1&t=Tub1wIzaPAXt2i86-1)

# Minerva the MATH LLM from Google
Minerva is an ancient-themed, state-of-the-art Language and Logic Model (LLM) developed by Google. Inspired by the wisdom of ancient mathematicians and logicians, Minerva combines their timeless insights with the power of modern technology to revolutionize the field of mathematics.

![Minerva Banner](minerva-banner.png)

[Join our Minerva Discord and contribute to this project or explore other mathematical wonders!](https://discord.gg/qUtxnK2NMf)

# Minerva: Unleashing the Secrets of Ancient Mathematics 🏛️🔢

<!-- ![Minerva Next Generation Open Source Language Model](/Minerva-banner.png) -->
Minerva is a groundbreaking language model that pushes the boundaries of mathematical understanding and problem-solving. Designed with an ancient math theme, Minerva embodies the spirit of renowned mathematicians such as Euclid, Pythagoras, and Archimedes. By harnessing their ancient wisdom, Minerva offers unparalleled capabilities in mathematical reasoning and exploration.

---

<div align="center">

[![GitHub issues](https://img.shields.io/github/issues/kyegomez/Minerva)](https://github.com/kyegomez/Minerva/issues) [![GitHub forks](https://img.shields.io/github/forks/kyegomez/Minerva)](https://github.com/kyegomez/Minerva/network) [![GitHub stars](https://img.shields.io/github/stars/kyegomez/Minerva)](https://github.com/kyegomez/Minerva/stargazers) [![GitHub license](https://img.shields.io/github/license/kyegomez/Minerva)](https://github.com/kyegomez/Minerva/blob/main/LICENSE)

</div>

<div align="center">

[![Share on Twitter](https://img.shields.io/twitter/url/https/twitter.com/cloudposse.svg?style=social&label=Share%20%40kyegomez/Minerva)](https://twitter.com/intent/tweet?text=Unleash%20the%20power%20of%20Minerva%20-%20the%20ancient-themed%20MATH%20LLM%20from%20Google!&url=https%3A%2F%2Fgithub.com%2Fkyegomez%2FMinerva) [![Share on Facebook](https://img.shields.io/badge/Share-%20facebook-blue)](https://www.facebook.com/sharer/sharer.php?u=https%3A%2F%2Fgithub.com%2Fkyegomez%2FMinerva) [![Share on LinkedIn](https://img.shields.io/badge/Share-%20linkedin-blue)](https://www.linkedin.com/shareArticle?mini=true&url=https%3A%2F%2Fgithub.com%2Fkyegomez%2FMinerva&title=&summary=&source=)

[![Share on Reddit](https://img.shields.io/badge/-Share%20on%20Reddit-orange)](https://www.reddit.com/submit?url=https%3A%2F%2Fgithub.com%2Fkyegomez%2FMinerva&title=Unleash%20the%20power%20of%20Minerva%20-%20the%20ancient-themed%20MATH%20LLM%20from%20Google!) [![Share on Hacker News](https://img.shields.io/badge/-Share%20on%20Hacker%20News-orange)](https://news.ycombinator.com/submitlink?u=https%3A%2F%2Fgithub.com%2Fkyegomez%2FMinerva&t=Unleash%20the%20power%20of%20Minerva%20-%20the%20ancient-themed%20MATH%20LLM%20from%20Google!) [![Share on Pinterest](https://img.shields.io/badge/-Share%20on%20Pinterest-red)](https://pinterest.com/pin/create/button/?url=https%3A%2F%2Fgithub.com%2Fkyegomez%2FMinerva&media=https%3A%2F%2Fexample.com%2Fimage.jpg&description=Unleash%20the%20power%20of%20Minerva%20-%20the%20ancient-themed%20MATH%20LLM%20from%20Google!) [![Share on WhatsApp](https://img.shields.io/badge/-Share%20on%20WhatsApp-green)](https://api.whatsapp.com/send?text=Unleash%20the%20power%20of%20Minerva%20-%20the%20ancient-themed%20MATH%20LLM%20from%20Google!%20%23Minerva%20%23AI%0A%0Ahttps%3A%2F%2Fgithub.com%2Fkyegomez%2FMinerva)

</div>

---




# Usage
There are two methods to use Minerva: one by installing via `pip` and the other by cloning the repository. For detailed instructions, please refer to the [Training SOP](DOCs/TRAINING.md).

# Method 1: Installation via `pip`

First, install Minerva by running the following command:

```shell
pip install Minerva-llm
```

Then, you can utilize Minerva in your Python code as shown below:

```python
import torch
from Minerva import Minerva, Train

# Example usage
x = torch.randint(0, 20000, (1, 1024))

Minerva(x)

# or train
Train()
```

## Method 2: Cloning the Repository

To get started with Method 2, follow these steps:

1. Clone the Minerva repository and navigate to the project directory:

```shell
git clone https://github.com/kyegomez/Minerva
cd Minerva
```

2. Install the required packages:

```shell
pip3 install -r requirements.txt
```

3. Run the training script:

```shell
cd Minerva
python3 training_distributed.py
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

```shell
python

3 Minerva/build_dataset.py --seed 42 --seq_len 8192 --hf_account "HUGGINGFACE APIKEY" --tokenizer "EleutherAI/gpt-neox-20b" --dataset_name "EleutherAI/the_pile_deduplicated"
```

# Inference

To perform inference with Minerva, use the `inference.py` script. Here's an example command:

```shell
python3 inference.py "My dog is very cute" --seq_len 256 --temperature 0.8 --filter_thres 0.9 --model "Minerva"
```

Please note that the Minerva model is not yet available on the PyTorch Hub, but we are actively working on making it accessible for easy integration into your projects.

## Roadmap 🗺️📍

1. **Training phase**: Train Minerva on a large-scale dataset to achieve state-of-the-art performance in various mathematical tasks and problem-solving.

2. **World-class inference infrastructure**: Establish a robust and efficient infrastructure for Minerva, incorporating techniques such as model quantization, distillation, and optimized serving frameworks. This ensures rapid and accurate mathematical inference.

3. **Continuous improvement**: Continuously fine-tune and expand Minerva's capabilities by incorporating new mathematical knowledge and adapting to emerging challenges and domains.

4. **Community-driven development**: Foster an open-source community around Minerva, encouraging contributions, improvements, and innovative use cases from mathematicians, researchers, and enthusiasts.

## Why Minerva? 🌠💡

Minerva sets itself apart with its unique ancient-themed approach, combining the wisdom of ancient mathematicians with modern deep learning techniques. Here are some reasons to choose Minerva:

- **Efficiency**: Minerva incorporates cutting-edge optimization techniques inspired by ancient mathematical principles, resulting in efficient training and inference.

- **Flexibility**: With its modular design, Minerva can adapt to various mathematical tasks and domains, making it a versatile choice for a wide range of applications.

- **Scalability**: Minerva's architecture is designed to scale with the ever-growing computational resources and data sizes, ensuring its continuous relevance in the world of mathematics.

- **Community-driven**: As an open-source project, Minerva thrives on contributions from the community, fostering collaboration, innovation, and continuous improvement.

Join us on this exciting journey to unlock the secrets of ancient mathematics and revolutionize the way we approach mathematical problem-solving! 🚀🌟