import numpy as np
import torch
from importlib.metadata import version
from torch.utils.data import Dataset, DataLoader
import tiktoken



# tokenizer = tiktoken.get_encoding("gpt2")
# text = "Hello, do you like tea? <|endoftext|> In the sunlit terraces of som"
# integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
# print(integers)
# strings = tokenizer.decode(integers)
# print(strings)
# with open("the-verdict.txt", "r", encoding="utf-8") as f:
#     raw_text = f.read()
# enc_text = tokenizer.encode(raw_text)
# print(len(enc_text))


class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.tokenizer = tokenizer
        self.input_ids = []
        self.target_ids = []
        token_ids = tokenizer.encode(txt) #A
        for i in range(0, len(token_ids) - max_length, stride): #B
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))
    def __len__(self): #C
        return len(self.input_ids)
    def __getitem__(self, idx): #D
        return self.input_ids[idx], self.target_ids[idx]
    
def create_dataloader_v1(txt, batch_size=4,
        max_length=256, stride=128, shuffle=True, drop_last=True):
    tokenizer = tiktoken.get_encoding("gpt2") #A
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride) #B
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
    return dataloader


with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
dataloader = create_dataloader_v1(
    raw_text, batch_size=1, max_length=4, stride=1, shuffle=False)
data_iter = iter(dataloader) #A
first_batch = next(data_iter)
# print(first_batch)
inputs, targets = next(data_iter)
# print("Inputs:\n", inputs)
# print("\nTargets:\n", targets)
torch.manual_seed(123)
vocab_size = 50257
output_dim=3
embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
# print(embedding_layer.weight)
# print(embedding_layer(torch.tensor([3])))
# input_ids= torch.tensor([2,5,3,1])
# print(embedding_layer(input_ids))
token_embeddings = embedding_layer(inputs)
# print(token_embeddings.shape)
max_length = 4
context_length = max_length
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
pos_embeddings = pos_embedding_layer(torch.arange(context_length))
# print(pos_embeddings.shape)
input_embeddings = token_embeddings + pos_embeddings
# print(input_embeddings.shape)