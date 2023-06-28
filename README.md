# Minerva: Solving Quantitative Reasoning Problems with Language Models

Minerva is a language model capable of solving mathematical and scientific questions using step-by-step reasoning. It was developed as a part of a Google Research project by Ethan Dyer and Guy Gur-Ari from the Blueshift Team. The goal is to enhance the quantitative reasoning abilities of language models. 

This repository contains the implementation details, model architecture, and the roadmap for the project.

## Table of Contents

1. [Model Architecture](#model-architecture)
2. [Implementation](#implementation)
3. [Roadmap](#roadmap)
4. [Share with Friends](#share-with-friends)
5. [Contact](#contact)

## Model Architecture

Minerva is built on the Pathways Language Model (PaLM), trained on a 118GB dataset of scientific papers from the arXiv preprint server and web pages that contain mathematical expressions. This dataset includes LaTeX, MathJax, and other mathematical typesetting formats. 

Unlike standard text cleaning procedures that often remove essential mathematical symbols and formatting, Minerva maintains this information to learn and communicate in standard mathematical notation.

The model is enhanced with techniques such as chain of thought or scratchpad prompting. This involves exposing Minerva to several step-by-step solutions before presenting a new question. 

Minerva also employs a majority voting system. When answering a question, multiple solutions are generated by sampling stochastically from all possible outputs. These solutions might have different steps but often reach the same final answer. Minerva uses majority voting on these sampled solutions, taking the most common result as the conclusive final answer.

## Implementation

To implement Minerva on your local system, follow the steps mentioned in the [Implementation Guide](IMPLEMENTATION_GUIDE.md). 

## Roadmap

Our goal is to continue improving Minerva's quantitative reasoning abilities. We're planning the following updates:

1. Incorporating a wider range of mathematical typesetting formats in training data.
2. Improving Minerva's capacity to handle complex multi-step mathematical problems.
3. Refining the majority voting system for more consistent results.
4. Enhancing the scratchpad prompting technique to provide more accurate steps in the solution process.

Check the [Project Roadmap](ROADMAP.md) for more details.

## Share with Friends

Feel free to share this project with anyone who might be interested. Just copy and paste the following Markdown text into your email, blog post, or any social media platform:

```
Check out [Minerva](https://github.com/kyegomez/Minerva) - a groundbreaking language model by Google Research, capable of solving mathematical and scientific questions using step-by-step reasoning.
```

## Contact

For questions or suggestions, please open an issue [here](https://github.com/kyegomez/Minerva/issues). We're always open to feedback and would love to hear from you! 

## License

Minerva is released under the [MIT License](LICENSE).
