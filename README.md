# [READY FOR TRAINING, help us with the strategy!](https://www.figma.com/file/pfaU8Nhyw0EdXuT6z4Hutw/Minerva-Strategy?type=whiteboard&node-id=0%3A1&t=Tub1wIzaPAXt2i86-1)

# Agora
Agora is an new open source Multi-Modality AI Research Organization devoted to advancing Humanity!

Since Minerva is ready to train Agora is actively seeking cloud providers or grant providers to train this all-new revolutionary model and release it open source, if you would like to learn more please email me at `kye@apac.ai`


![Agora banner](agora-banner.png)

[Join our Agora discord and contribute to this project or 40+ others!](https://discord.gg/qUtxnK2NMf)


# Minerva: Ultra-Fast and Ultra-Intelligent SOTA Language Model 🚀🌌

<!-- ![Minerva Next Generation Open Source Language Model](/Minerva-banner.png) -->
Minerva is a state-of-the-art language model that pushes the boundaries of natural language understanding and generation. Designed for high performance and efficiency, Minerva is built upon advanced techniques that make it a strong contender against the likes of OpenAI's GPT-4 and PALM.

---

<div align="center">

[![GitHub issues](https://img.shields.io/github/issues/kyegomez/Minerva)](https://github.com/kyegomez/Minerva/issues) [![GitHub forks](https://img.shields.io/github/forks/kyegomez/Minerva)](https://github.com/kyegomez/Minerva/network) [![GitHub stars](https://img.shields.io/github/stars/kyegomez/Minerva)](https://github.com/kyegomez/Minerva/stargazers) [![GitHub license](https://img.shields.io/github/license/kyegomez/Minerva)](https://github.com/kyegomez/Minerva/blob/main/LICENSE)

</div>

<div align="center">

[![Share on Twitter](https://img.shields.io/twitter/url/https/twitter.com/cloudposse.svg?style=social&label=Share%20%40kyegomez/Minerva)](https://twitter.com/intent/tweet?text=Check%20out%20this%20amazing%20AI%20project:%20Minerva&url=https%3A%2F%2Fgithub.com%2Fkyegomez%2FMinerva) [![Share on Facebook](https://img.shields.io/badge/Share-%20facebook-blue)](https://www.facebook.com/sharer/sharer.php?u=https%3A%2F%2Fgithub.com%2Fkyegomez%2FMinerva) [![Share on LinkedIn](https://img.shields.io/badge/Share-%20linkedin-blue)](https://www.linkedin.com/shareArticle?mini=true&url=https%3A%2F%2Fgithub.com%2Fkyegomez%2FMinerva&title=&summary=&source=)

[![Share on Reddit](https://img.shields.io/badge/-Share%20on%20Reddit-orange)](https://www.reddit.com/submit?url=https%3A%2F%2Fgithub.com%2Fkyegomez%2FMinerva&title=Minerva%20-%20the%20next%20generation%20AI%20shields) [![Share on Hacker News](https://img.shields.io/badge/-Share%20on%20Hacker%20News-orange)](https://news.ycombinator.com/submitlink?u=https%3A%2F%2Fgithub.com%2Fkyegomez%2FMinerva&t=Minerva%20-%20the%20next%20generation%20AI%20shields) [![Share on Pinterest](https://img.shields.io/badge/-Share%20on%20Pinterest-red)](https://pinterest.com/pin/create/button/?url=https%3A%2F%2Fgithub.com%2Fkyegomez%2FMinerva&media=https%3A%2F%2Fexample.com%2Fimage.jpg&description=Minerva%20-%20the%20next%20generation%20AI%20shields) [![Share on WhatsApp](https://img.shields.io/badge/-Share%20on%20WhatsApp-green)](https://api.whatsapp.com/send?text=Check%20out%20Minerva%20-%20the%20next%20generation%20AI%20shields%20%23Minerva%20%23AI%0A%0Ahttps%3A%2F%2Fgithub.com%2Fkyegomez%2FMinerva)

</div>

---




# Usage
There are 2 methods to use Minerva, 1 by `pip install Minerva-llm` and the other by `git clone`. [Head over to the Training SOP for more](DOCs/TRAINING.md)

# Method1
First `pip install Minerva-llm` then

```python
import torch
from Minerva import Minerva, Train


x = torch.randint(0, 20000, (1, 1024))

Minerva(x)

# or train

Train()

```


## Method 2

Get started:

1. Clone the repository and install the required packages.


```
git clone https://github.com/kyegomez/Minerva
cd Minerva
pip3 install -r requirements.txt
cd Minerva
python3 training_distributed.py
```

# Training

First:

`Accelerate Config`

Enable Deepspeed 3: 

`Accelerate launch train_distributed_accelerate.py`

# Environment variables

* `ENTITY_NAME` ==> Your wandb project name

* `OUTPUT_DIR` ==> Where you want the weights to go when it's finished training for example inside the root directory you can do something like: `./weights` and it'll create a folder called weights INSIDE of the Minerva folder

## Dataset building building

Data
You can preprocess a different dataset in a way similar to the C4 dataset used during training by running the build_dataset.py script. This will pre-tokenize, chunk the data in blocks of a specified sequence length, and upload to the Huggingface hub. For example:

```python3 Minerva/build_dataset.py --seed 42 --seq_len 8192 --hf_account "HUGGINGFACE APIKEY" --tokenizer "EleutherAI/gpt-neox-20b" --dataset_name "EleutherAI/the_pile_deduplicated"```



# Inference

```python3 inference.py "My dog is very cute" --seq_len 256 --temperature 0.8 --filter_thres 0.9 --model "Minerva"``` 

Not yet we need to submit model to pytorch hub



## Roadmap 🗺️📍

1. **Training phase**: Train Minerva on a large-scale dataset to achieve SOTA performance in various natural language processing tasks.

2. **World-class inference infrastructure**: Establish a robust and efficient infrastructure that leverages techniques such as:

   - Model quantization: Reduce memory and computational requirements without significant loss in performance.
   - Distillation: Train smaller, faster models that retain the knowledge of the larger model.
   - Optimized serving frameworks: Deploy Minerva using efficient serving frameworks, such as NVIDIA Triton or TensorFlow Serving, for rapid inference.

3. **Continuous improvement**: Continuously fine-tune Minerva on diverse data sources and adapt it to new tasks and domains.

4. **Community-driven development**: Encourage open-source contributions, including pre-processing improvements, advanced training techniques, and novel use cases.

## Why Minerva? 🌠💡

Minerva can potentially be finetuned with 100k+ token sequence length.
Minerva is a state-of-the-art language model that leverages advanced techniques to optimize its performance and efficiency. Some of these techniques include alibi positional bias, rotary position encodings (xpos), flash attention, and deep normalization (deepnorm). Let's explore the benefits of these techniques and provide some usage examples.


### Flash Attention

Flash attention speeds up the self-attention mechanism by reducing the number of attention computations. It accelerates training and inference while maintaining a high level of performance.

Usage example:

```python
attn_layers = Decoder(
    ...
    attn_flash=True,
    ...
)
```

Usage example:

```python
attn_layers = Decoder(
    ...
    deepnorm=True,
    ...
)
```

### Deep Normalization (deepnorm)

Deep normalization is a technique that normalizes the activations within a layer, helping with training stability and convergence. It allows the model to better learn complex patterns and generalize to unseen data.

# Minerva Principles
- **Efficiency**: Minerva incorporates cutting-edge optimization techniques, such as attention flashing, rotary position encodings, and deep normalization, resulting in efficient training and inference.

- **Flexibility**: The modular design of Minerva allows for easy adaptation to various tasks and domains, making it a versatile choice for a wide range of applications.

- **Scalability**: Minerva's architecture is designed to scale with the ever-growing computational resources and data sizes, ensuring its continuous relevance in the NLP landscape.

- **Community-driven**: As an open-source project, Minerva thrives on contributions from the community, fostering an environment of collaboration, innovation, and continuous improvement.

Join us on this exciting journey to create a powerful, efficient, and intelligent language model that will revolutionize the NLP landscape! 🚀🌟

