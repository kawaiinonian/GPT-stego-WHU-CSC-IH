import torch
import torch.nn.functional as F
import os
import argparse
from tqdm import trange
from transformers import GPT2LMHeadModel
from tokenizations import tokenization_bert
import numpy as np
import scipy.stats


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits


def near(alist, anum):
	up = len(alist) - 1
	if up == 0:
		return 0
	bottom = 0
	while up - bottom > 1:
		index = int((up + bottom)/2)
		if alist[index] < anum:
			up = index
		elif alist[index] > anum:
			bottom = index
		else:
			return index
	if up - bottom == 1:
		if alist[bottom] - anum < anum - up:
			index = bottom
		else:
			index = up
	return index

def bits2int(bits):
	res = 0
	for i, bit in enumerate(bits):
		res += bit*(2**i)
	return res

def int2bits(inp, num_bits):
	if num_bits == 0:
		return []
	strlist = ('{0:0%db}'%num_bits).format(inp)
	return [strval for strval in reversed(strlist)]

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

def packet_sampling(logits, bit_stream, bit_index, device):
    prob = torch.exp(logits)
    prob = prob / prob.sum()
    prob, indices = prob.sort(descending=True)
    # start recursion
    bit_tmp = 0
    while prob[0] <= 0.5:
        # embedding bit
        bit = 1
        while (1 / 2 ** (bit + 1)) > prob[0]:
            bit += 1
        mean = 1 / 2 ** bit
        # dp
        prob = prob.tolist()
        indices = indices.tolist()
        result = []
        for i in range(2 ** bit):
            result.append([[], []])
        for i in range(2 ** bit - 1):
            result[i][0].append(prob[0])
            result[i][1].append(indices[0])
            del (prob[0])
            del (indices[0])
            while sum(result[i][0]) < mean:
                delta = mean - sum(result[i][0])
                index = near(prob, delta)
                if prob[index] - delta < delta:
                    result[i][0].append(prob[index])
                    result[i][1].append(indices[index])
                    del (prob[index])
                    del (indices[index])
                else:
                    break
            mean = sum(prob) / (2 ** bit - i - 1)
        result[2 ** bit - 1][0].extend(prob)
        result[2 ** bit - 1][1].extend(indices)
        # read secret message
        if bit_index + bit_tmp > len(bit_stream):
             bit_embed = [0]
        elif bit_index + bit + bit_tmp > len(bit_stream):
            bit_embed = [int(_) for _ in bit_stream[bit_index + bit_tmp:]]
        else:
            bit_embed = [int(_) for _ in bit_stream[bit_index + bit_tmp:bit_index + bit_tmp + bit]]
        int_embed = bits2int(bit_embed)
        # updating 
        prob = torch.FloatTensor(result[int_embed][0]).to(device)
        indices = torch.LongTensor(result[int_embed][1]).to(device)
        prob = prob / prob.sum()
        prob, _ = prob.sort(descending=True)
        indices = indices[_]
        bit_tmp += bit
    return prob, indices, bit_index + bit_tmp

def do_generate(hidden_bits, bit_index, length, n_ctx, tokenizer, model, device, context):
    context = torch.tensor(context, dtype=torch.long, device=device)
    context = context.unsqueeze(0)
    generated = context
    with torch.no_grad():
        for _ in trange(length):
            inputs = {'input_ids': generated[0][-(n_ctx - 1):].unsqueeze(0)}
            outputs = model(
                **inputs)  # Note: we could also use 'past' with GPT-2/Transfo-XL/XLNet (cached hidden-states)
            next_token_logits = outputs[0][0, -1, :]
            next_token_logits[tokenizer.convert_tokens_to_ids('[UNK]')] = -float('Inf')
            next_token_logits[tokenizer.convert_tokens_to_ids('[CLS]')] = -float('Inf')
            next_token_logits[tokenizer.convert_tokens_to_ids('##')] = -float('Inf')
            prob, indices, bit_index = packet_sampling(next_token_logits, hidden_bits, bit_index, device)
            next_token = indices[int(torch.multinomial(prob, 1))].view([1])
            generated = torch.cat((generated, next_token.unsqueeze(0)), dim=1)
    return generated

def generater(title: str, article_num: int, hidden_bits: str, length: int, batch_num:int):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ret = []
    bit_index = 0

    tokenizer = tokenization_bert.BertTokenizer(vocab_file='cache/vocab.txt')
    model = GPT2LMHeadModel.from_pretrained('model/')
    model.to(device)
    model.eval()
    n_ctx = model.config.n_ctx
    for i in range(article_num):
        context_tokens = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(title))
        generate_text = do_generate(
            hidden_bits = hidden_bits, 
            bit_index = bit_index,
            length = length, n_ctx = n_ctx,
            model = model,
            device = device,
            tokenizer = tokenizer,
            context = context_tokens)
        generate_text = generate_text.tolist()[0]
        text = tokenizer.convert_ids_to_tokens(generate_text)

        for i, item in enumerate(text):
            if item == '[MASK]':
                text[i] = ''
            # if item == '[CLS]':
            #     text[i] = '\n'
            if item == '[SEP]':
                text[i] = '\n'
        text = "".join(text).replace('##', '').strip()
        ret.append(text)
    return text

def packet_sampling_for_extract(logits, next_token, device):
    prob = torch.exp(logits)
    prob = prob / prob.sum()
    prob, indices = prob.sort(descending=True)
    # start recursion
    bit_tmp = 0
    ret = []
    while prob[0] <= 0.5:
        # embedding bit
        bit = 1
        while (1 / 2 ** (bit + 1)) > prob[0]:
            bit += 1
        mean = 1 / 2 ** bit
        # dp
        prob = prob.tolist()
        indices = indices.tolist()
        result = []
        for i in range(2 ** bit):
            result.append([[], []])
        for i in range(2 ** bit - 1):
            result[i][0].append(prob[0])
            result[i][1].append(indices[0])
            del (prob[0])
            del (indices[0])
            while sum(result[i][0]) < mean:
                delta = mean - sum(result[i][0])
                index = near(prob, delta)
                if prob[index] - delta < delta:
                    result[i][0].append(prob[index])
                    result[i][1].append(indices[index])
                    del (prob[index])
                    del (indices[index])
                else:
                    break
            mean = sum(prob) / (2 ** bit - i - 1)
        result[2 ** bit - 1][0].extend(prob)
        result[2 ** bit - 1][1].extend(indices)
        choice = 0
        for i in range(2 ** bit):
            if next_token in result[i][1]:
                ret.append("".join(int2bits(i, bit)))
                choice = i
                break
        prob = torch.FloatTensor(result[choice][0]).to(device)
        indices = torch.LongTensor(result[choice][1]).to(device)
        prob = prob / prob.sum()
        prob, _ = prob.sort(descending=True)
        indices = indices[_]
    return "".join(ret)

def extract(text, length, title):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bit_index = 0
    text_length = len(text)

    tokenizer = tokenization_bert.BertTokenizer(vocab_file='cache/vocab.txt')
    model = GPT2LMHeadModel.from_pretrained('model/')
    model.to(device)
    model.eval()
    n_ctx = model.config.n_ctx
    text = text.replace('\n', '[SEP]')
    context = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(title))
    context = torch.tensor(context, dtype=torch.long, device=device)
    context = context.unsqueeze(0)
    text = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))
    text = torch.tensor(text, dtype=torch.long, device=device)
    # text = text.unsqueeze(0)
    text_index = len(title)
    bit_stream = []
    while bit_index < length:
        inputs = {'input_ids': context[0][-(n_ctx - 1):].unsqueeze(0)}
        outputs = model(
            **inputs)  # Note: we could also use 'past' with GPT-2/Transfo-XL/XLNet (cached hidden-states)
        next_token_logits = outputs[0][0, -1, :]
        next_token_logits[tokenizer.convert_tokens_to_ids('[UNK]')] = -float('Inf')
        next_token_logits[tokenizer.convert_tokens_to_ids('[CLS]')] = -float('Inf')
        next_token_logits[tokenizer.convert_tokens_to_ids('##')] = -float('Inf')
        bit_extract = packet_sampling_for_extract(next_token_logits, text[text_index], device)
        next_token = text[text_index].view([1])
        text_index += 1
        context = torch.cat((context, next_token.unsqueeze(0)), dim=1)
        bit_index += len(bit_extract)
        bit_stream.append(bit_extract)
    return "".join(bit_stream)

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


# stream = 'CrazyThurdayVme50'
# title = '雨一直下，风一直刮'
# stream_b = "".join([format(ord(i), '08b') for i in stream])
# print(stream_b)
# text = generater(title, 1, stream_b)
# print(text)
# stream_extract = extract(text[0], len(stream_b), title)
# print(stream_extract)
# print(bin2str(stream_extract))
# with open('data.txt', mode='w') as f:
#     f.write(str(stream_b))
#     f.write(str(text))
#     f.write(str(stream_extract))
#     f.write(str(bin2str(stream_extract)))
