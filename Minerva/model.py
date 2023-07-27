import torch

import torch.nn as nn
from palm_rlhf_pytorch import PaLM
from transformer import AutoTokenizer
import bitsandbytes as bnb

from Minerva.embedding import PositionalEmbedding

class MinervaTokenizer:
    def __init__(self):
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                    "EleutherAI/gpt-neox-20b",
                    eos_token="<eos>",
                    pad_token="<pad>",
                    extra_ids=0,
                    model_max_length=8192
                )
            
        except Exception as e:
            print(f"Error init in tokenizer: {e}")

    def tokenize_texts(self, texts):
        try:
            texts = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True).input_ids
            return texts, texts
        except Exception as e:
            print(f"Error tokenizing texts: {e}")


class Minerva(nn.Module):
    def __init__(self):
        super(Minerva, self).__init__()
        try:
            self.embed = bnb.nn.modules.Embedding(
                320002,
                2048,
                padding_idx=1
            )
            
            try:
                self.embed_positions = PositionalEmbedding(2048, 2048, 1)
            except Exception as e:
                print(str(e))

            torch.nn.init.normal_(
                self.output_projection.weight, mean=0, std=2048**-0.5
            )

            self.decoder = PaLM(
                num_tokens=50304,
                dim=2048,
                depth=16,
                dim_head=128,
                heads=8,
                flash_attn=True,
                qk_rmsnorm=False
            )
    
        except Exception as e:
            print(f"Error initializing components; {e}")

    
    def forward(self, text_tokens):
        try:
            model_input = self.decoder(text_tokens)
            output = self.decoder(model_input, passed_x=model_input)[0]

            return output
        except Exception as e:
            print(f"Error during forward pass: {e}")
            return None

