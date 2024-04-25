
import time
import argparse
import struct
import numpy as np
from collections import OrderedDict
import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import ViTForImageClassification


class VitConfig:
    depths: list = [3, 4, 6, 3]
    embedding_size: int = 64
    hidden_sizes: list = [256, 512, 1024, 2048]
    num_channels: int = 3

class Vit(nn.Module):
    @classmethod
    def from_pretrained(cls, model_type):
        print("loading weights from pretrained resnet: %s" % model_type)

        model_hf = ViTForImageClassification.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()
        # print(sd_hf.keys())
        return model_hf

def write_fp32(tensor, file):
    file.write(tensor.detach().numpy().astype("float32").tobytes())

def write_model(model, filename):
    print(f"write model to {filename}, the keys of model is {len(model.state_dict())}")
    config = model.config
    with open(filename, "wb") as file:
        file.write(struct.pack("i", 20240423)) # magic
        file.write(struct.pack("i", config.encoder_stride))
        file.write(struct.pack("i", config.hidden_size))
        file.write(struct.pack("i", config.image_size))
        file.write(struct.pack("i", config.intermediate_size))
        file.write(struct.pack("i", config.num_attention_heads))
        file.write(struct.pack("i", config.num_channels))
        file.write(struct.pack("i", config.num_hidden_layers))
        file.write(struct.pack("i", config.patch_size))
        file.write(struct.pack("i", 1000)) # num_labels

        sd = model.state_dict()

        # embedder 4
        write_fp32(sd["vit.embeddings.patch_embeddings.projection.weight"], file) # [hidden_size, num_channels, patch_size, patch_size]
        # print(sd["vit.embeddings.patch_embeddings.projection.weight"].shape)
        write_fp32(sd["vit.embeddings.patch_embeddings.projection.bias"], file) # [hidden_size]
        # print(sd["vit.embeddings.patch_embeddings.projection.bias"].shape) # 
        write_fp32(sd["vit.embeddings.cls_token"], file) # [1, 1, 768]  [1, 1, hidden_size]
        # print(sd["vit.embeddings.cls_token"].shape)
        write_fp32(sd["vit.embeddings.position_embeddings"], file) # [1, 197, 768]  [1, num_patches + 1, hidden_size]
        # print(sd["vit.embeddings.position_embeddings"].shape)
    
        #  16 * 12 = 192
        for i in range(config.num_hidden_layers): # num_hidden_layers * hidden_size * all_head_size 
            write_fp32(sd[f"vit.encoder.layer.{i}.attention.attention.query.weight"], file) # [768, 768]
        for i in range(config.num_hidden_layers): # num_hidden_layers * all_head_size 
            write_fp32(sd[f"vit.encoder.layer.{i}.attention.attention.query.bias"], file) # [768]
        for i in range(config.num_hidden_layers): # num_hidden_layers * hidden_size * all_head_size 
            write_fp32(sd[f"vit.encoder.layer.{i}.attention.attention.key.weight"], file) # [768, 768]
        for i in range(config.num_hidden_layers): # num_hidden_layers * all_head_size 
            write_fp32(sd[f"vit.encoder.layer.{i}.attention.attention.key.bias"], file) # [768]
        for i in range(config.num_hidden_layers): # num_hidden_layers * hidden_size * all_head_size 
            write_fp32(sd[f"vit.encoder.layer.{i}.attention.attention.value.weight"], file) # [768, 768]
        for i in range(config.num_hidden_layers): # num_hidden_layers * all_head_size 
            write_fp32(sd[f"vit.encoder.layer.{i}.attention.attention.value.bias"], file) # [768]
        for i in range(config.num_hidden_layers): # num_hidden_layers * hidden_size * hidden_size 
            write_fp32(sd[f"vit.encoder.layer.{i}.attention.output.dense.weight"], file) # [768, 768]
        for i in range(config.num_hidden_layers): # num_hidden_layers * hidden_size
            write_fp32(sd[f"vit.encoder.layer.{i}.attention.output.dense.bias"], file) # [768]
        for i in range(config.num_hidden_layers): # num_hidden_layers * hidden_size * intermediate_size
            write_fp32(sd[f"vit.encoder.layer.{i}.intermediate.dense.weight"], file) # [768, 3072]
        for i in range(config.num_hidden_layers): # num_hidden_layers * intermediate_size;
            write_fp32(sd[f"vit.encoder.layer.{i}.intermediate.dense.bias"], file) # [3072]
        for i in range(config.num_hidden_layers): # num_hidden_layers * intermediate_size * hidden_size
            write_fp32(sd[f"vit.encoder.layer.{i}.output.dense.weight"], file) # [3072, 768]
        for i in range(config.num_hidden_layers): # num_hidden_layers * hidden_size
            write_fp32(sd[f"vit.encoder.layer.{i}.output.dense.bias"], file) # [768]
        for i in range(config.num_hidden_layers): # num_hidden_layers * hidden_size
            write_fp32(sd[f"vit.encoder.layer.{i}.layernorm_before.weight"], file) # [768]
        for i in range(config.num_hidden_layers): # num_hidden_layers * hidden_size
            write_fp32(sd[f"vit.encoder.layer.{i}.layernorm_before.bias"], file) # [768]
        for i in range(config.num_hidden_layers): # num_hidden_layers * hidden_size
            write_fp32(sd[f"vit.encoder.layer.{i}.layernorm_after.weight"], file) # [768]
        for i in range(config.num_hidden_layers): # num_hidden_layers * hidden_size
            write_fp32(sd[f"vit.encoder.layer.{i}.layernorm_after.bias"], file) # [768]

        # 4
        write_fp32(sd["vit.layernorm.weight"], file) # hidden_size [768]
        write_fp32(sd["vit.layernorm.bias"], file) # hidden_size [768]
        write_fp32(sd["classifier.weight"], file) # hidden_size * num_labels [768, 1000]
        write_fp32(sd["classifier.bias"], file) # num_labels [1000]

        # 4 + 192 + 4 = 200

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    device = "cpu"
    # load the GPT-2 model weights
    model = Vit.from_pretrained("google/vit-base-patch16-224")
    # print(model)
    config = model.config
    # print(config)
    # with open("id2label.bin", "wb") as file:
    #     id2label = config.id2label
    #     id2label = OrderedDict(id2label)
        
    #     # print(len(id2label[7]), id2label[7])
    #     for k, v in id2label.items():
    #         fmt = f"{len(v)}s"
    #         # print(fmt)
    #         bk = struct.pack("i", len(v))
    #         bv = struct.pack(f"{len(v)}s", bytes(v, 'utf-8'))
    #         file.write(bk)
    #         file.write(bv)
    filename = "vit-base-patch16-224.bin"
    write_model(model, filename)
