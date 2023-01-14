from transformers import GPT2LMHeadModel
from tokenizations import tokenization_bert
import torch
import numpy as np
import scipy.stats

def get_ppl(sent):

    tokenizer = tokenization_bert.BertTokenizer(vocab_file='cache/vocab.txt')
    model = GPT2LMHeadModel.from_pretrained('model/')
    model.eval()
    tokens = tokenizer.tokenize(sent)
    tensor_input = torch.tensor([tokenizer.convert_tokens_to_ids(tokens)])

    loss = model(tensor_input, labels=tensor_input)[0]
    ppl = np.exp(loss.data.item())

    return ppl

def get_kl(sent1, sent2):

    tokenizer = tokenization_bert.BertTokenizer(vocab_file='cache/vocab.txt')

    tokens1 = tokenizer.tokenize(sent1)
    tokens2 = tokenizer.tokenize(sent2)
    if len(tokens1) > len(tokens2):
        # tensor_input1 = torch.tensor([tokenizer.encode(sent1, max_length=len(tokens2), truncation=True)], dtype=torch.float64)
        # tensor_input2 = torch.tensor([tokenizer.encode(sent2, max_length=len(tokens2), truncation=True)], dtype=torch.float64)
        input1 = tokenizer.encode(sent1, max_length=len(tokens2), truncation=True)
        input2 = tokenizer.encode(sent2, max_length=len(tokens2), truncation=True)
    else:
        # tensor_input1 = torch.tensor([tokenizer.encode(sent1, max_length=len(tokens1), truncation=True)], dtype=torch.float64)
        # tensor_input2 = torch.tensor([tokenizer.encode(sent2, max_length=len(tokens1), truncation=True)], dtype=torch.float64)
        input1 = tokenizer.encode(sent1, max_length=len(tokens1), truncation=True)
        input2 = tokenizer.encode(sent2, max_length=len(tokens1), truncation=True)

    # kl = torch.nn.functional.kl_div(tensor_input1.log(), tensor_input2, reduction='sum')
    kl = scipy.stats.entropy(input1, input2)
    return kl

def stream_evaluate(input: str , target: str):
    if len(input) < len(target):
        print("extract error")
        return 0
    j = 0
    for i in range(len(target)):
        if input[i] == target[i]:
            j += 1
    j += len(input) - len(target)
    return j/len(input)

def bin2str(string):
    buffer = []
    ret = []
    for i in range(len(string)):
        if i%8 == 0 and i != 0:
            byte = "".join(buffer)
            buffer = []
            byte = chr(int(byte, base=2))
            ret.append(byte)
        buffer.append(string[i])
    byte = "".join(buffer)
    byte = chr(int(byte, base=2))
    ret.append(byte)
    ret = "".join(ret)
    return ret
