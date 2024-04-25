from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import requests
import torch

url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)

processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

inputs = processor(images=[image, image], return_tensors="pt")

# print(inputs["pixel_values"].shape)
outputs = model(**inputs)
logits = outputs.logits
# print(logits.softmax(-1).max(-1))
# model predicts one of the 1000 ImageNet classes
# predicted_class_idx = logits.argmax(-1).item()
# print("Predicted class:", model.config.id2label[predicted_class_idx])

inputs = inputs["pixel_values"]
# print(inputs)

# outputs = model.vit.embeddings.patch_embeddings.projection(inputs) # [1, 768, 14, 14]
# print(outputs.shape)
# outputs = outputs.flatten(2) # [1, 768, 196]
# print(outputs.shape)
# outputs = outputs.transpose(1, 2) # [1, 196, 768]
# print(outputs.shape)
# # outputs = 
# print(model.vit.embeddings.cls_token.shape) # ([1, 1, 768])
# print(model.vit.embeddings.cls_token.expand(2, -1, -1).shape) # ([B, 1, 768])

embeddings = model.vit.embeddings.patch_embeddings.projection(inputs).flatten(2).transpose(1, 2) # [1, 196, 768]
# print(embeddings.shape)
cls_tokens = model.vit.embeddings.cls_token.expand(2, -1, -1) # 2, 1, 768]
# print(cls_tokens.shape)
embeddings = torch.cat((cls_tokens, embeddings), dim=1) # [2, 197, 768]
# print(embeddings.shape)
position_embeddings = model.vit.embeddings.position_embeddings # [1, 197, 768]
# print(position_embeddings.shape)
embeddings = embeddings + position_embeddings # [2, 197, 768]
# print(embeddings)
embeddings = model.vit.encoder(embeddings)[0]
# print(embeddings)
embeddings = model.vit.layernorm(embeddings)
# print(embeddings[:, 0, :].shape)
embeddings = model.classifier(embeddings[:, 0, :])
# print(embeddings)
embeddings = embeddings.softmax(-1)
# print(embeddings)
embeddings = embeddings.argmax(-1)
print(embeddings)
# embeddings = model.vit.encoder.layer[0](embeddings)[0]
# # print(embeddings)
# embeddings = model.vit.encoder.layer[1](embeddings)[0]
# # print(embeddings)
# embeddings = model.vit.encoder.layer[2](embeddings)[0]
# # print(embeddings)
# embeddings = model.vit.encoder.layer[3](embeddings)[0]
# # print(embeddings)
# embeddings = model.vit.encoder.layer[4](embeddings)[0]
# # print(embeddings)
# embeddings = model.vit.encoder.layer[5](embeddings)[0]
# # print(embeddings)
# embeddings = model.vit.encoder.layer[6](embeddings)[0]
# # print(embeddings)
# embeddings = model.vit.encoder.layer[7](embeddings)[0]
# # print(embeddings)
# embeddings = model.vit.encoder.layer[8](embeddings)[0]
# # print(embeddings)
# embeddings = model.vit.encoder.layer[9](embeddings)[0]
# # print(embeddings)
# embeddings = model.vit.encoder.layer[10](embeddings)[0]
# # print(embeddings)
# embeddings = model.vit.encoder.layer[11](embeddings)[0]
# print(embeddings)
# hidden_states = model.vit.encoder.layer[1].layernorm_before(embeddings)
# print(hidden_states)

# context_layer, attention_probs = model.vit.encoder.layer[1].attention.attention(hidden_states, output_attentions=True)
# # print(context_layer)
# # # print(attention_probs)
# attention_output = model.vit.encoder.layer[1].attention.output(context_layer, None)
# # print(attention_output)
# hidden_states = attention_output + embeddings
# # print(hidden_states)
# layer_output = model.vit.encoder.layer[1].layernorm_after(hidden_states)
# # print(layer_output)
# layer_output = model.vit.encoder.layer[1].intermediate.dense(layer_output)
# # print(layer_output)
# layer_output = model.vit.encoder.layer[1].intermediate.intermediate_act_fn(layer_output)
# # print(layer_output)
# layer_output = model.vit.encoder.layer[1].output.dense(layer_output)
# # print(layer_output)
# layer_output = layer_output + hidden_states
# print(layer_output)
# layer_output = model.vit.encoder.layer[0](embeddings)
# # print(layer_output)
# layer_output = model.vit.encoder.layer[1](layer_output)
# print(layer_output)
# def transpose_for_scores(x: torch.Tensor) -> torch.Tensor:
#     new_x_shape = x.size()[:-1] + (12, 64)
#     x = x.view(new_x_shape)
#     return x.permute(0, 2, 1, 3)

# hidden_states = model.vit.encoder.layer[1].layernorm_before(embeddings)
# querys = model.vit.encoder.layer[1].attention.attention.query(hidden_states)
# # print(querys)
# # query_layer = transpose_for_scores(querys)
# # # print(querys)
# keys = model.vit.encoder.layer[1].attention.attention.key(hidden_states)
# # print(keys)
# # key_layer = transpose_for_scores(keys)
# # # print(keys)
# values = model.vit.encoder.layer[1].attention.attention.value(hidden_states)
# print(values)
# values = transpose_for_scores(values)
# # print(values)
# attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
# # print(attention_scores)
# import math
# import torch.nn as nn
# attention_scores = attention_scores / math.sqrt(64)
# # print(attention_scores)
# attention_probs = nn.functional.softmax(attention_scores, dim=-1)
# attention_probs = attention_scores.max(-1)
# print(attention_probs.values.shape)
# print(model.vit.embeddings.patch_embeddings.projection.weight) # [768, 3, 16, 16]
# print(model.vit.embeddings.patch_embeddings.projection.bias) # [768]
# print(model.vit.embeddings.cls_token.shape) # [1, 1, 768]
# print(model.vit.embeddings.position_embeddings.shape) # [1, 197, 768]

# print(model.vit.encoder.layer[0].attention.attention.query.weight)
# print(model.vit.encoder.layer[0].attention.attention.query.bias) # [768]
# print(model.vit.encoder.layer[0].layernorm_before.weight.shape)
# print(model.vit.encoder.layer[0].layernorm_before.bias.shape)
# print(model.vit.encoder.layer[0].layernorm_after.weight)
# print(model.vit.encoder.layer[0].layernorm_after.bias)