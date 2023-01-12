import torch
import torch.nn.functional as F
from tqdm import trange
from transformers import GPT2LMHeadModel
from tokenizations import tokenization_bert
import numpy as np
from Hamming import hamming_encode, hamming_decode

temperature = 0.95
using_cover = []

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
    flag = False
    with torch.no_grad():
        for i in trange(length):
            inputs = {'input_ids': generated[0][-(n_ctx - 1):].unsqueeze(0)}
            outputs = model(
                **inputs)  # Note: we could also use 'past' with GPT-2/Transfo-XL/XLNet (cached hidden-states)
            next_token_logits = outputs[0][0, -1, :]
            next_token_logits = next_token_logits * temperature
            next_token_logits[tokenizer.convert_tokens_to_ids('[UNK]')] = -float('Inf')
            next_token_logits[tokenizer.convert_tokens_to_ids('[CLS]')] = -float('Inf')
            next_token_logits[tokenizer.convert_tokens_to_ids('##')] = -float('Inf')
            prob, indices, bit_index = packet_sampling(next_token_logits, hidden_bits, bit_index, device)
            if bit_index >= len(hidden_bits) and flag == False:
                using_cover.append(i+1)
                flag = True
            next_token = indices[int(torch.multinomial(prob, 1))].view([1])
            generated = torch.cat((generated, next_token.unsqueeze(0)), dim=1)
    return generated

def generater(title: str, article_num: int, hidden_bits: str, length = 256):
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
        with open('./debug_generate.txt', mode='w') as f:
            f.write(str(generate_text))
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
    return ret

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
    tokenizer = tokenization_bert.BertTokenizer(vocab_file='cache/vocab.txt')
    model = GPT2LMHeadModel.from_pretrained('model/')
    model.to(device)
    model.eval()
    n_ctx = model.config.n_ctx
    text = text.replace('\n', '[SEP]')    
    # text = text.replace('\n', '[CLS]')
    context = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(title))
    context = torch.tensor(context, dtype=torch.long, device=device)
    context = context.unsqueeze(0)
    text = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))
    text = torch.tensor(text, dtype=torch.long, device=device)
    with open('./debug_extract.txt', mode='w') as f:
        text_list = text.tolist()
        f.write(str(text_list))
    # text = text.unsqueeze(0)
    text_index = len(title)
    bit_stream = []
    while bit_index < length:
        inputs = {'input_ids': context[0][-(n_ctx - 1):].unsqueeze(0)}
        outputs = model(
            **inputs)  # Note: we could also use 'past' with GPT-2/Transfo-XL/XLNet (cached hidden-states)
        next_token_logits = outputs[0][0, -1, :]
        next_token_logits = next_token_logits * temperature
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

# stream = 'CrazyThursdayVm'
# title = '我很想念你，但是'
# stream_b = "".join([format(ord(i), '08b') for i in stream])
# stream_b = hamming_encode(stream_b)
# text = generater(title, 1, stream_b)
# print(text)
# stream_extract = extract(text[0], len(stream_b), title)
# stream_extract = hamming_decode(stream_extract)
# print(bin2str(stream_extract))

# print(using_cover)
# print(np.array(using_cover).sum()/len(using_cover))