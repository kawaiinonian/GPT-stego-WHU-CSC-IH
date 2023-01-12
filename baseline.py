import torch
import torch.nn.functional as F
from tqdm import trange
from transformers import GPT2LMHeadModel
from tokenizations import tokenization_bert

def bits2int(bits):
	res = 0
	for i, bit in enumerate(bits):
		res += bit*(2**i)
	return res

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


def top_k_sample(logits, bit_stream, bit_index, k):
    prob = torch.exp(logits)
    prob = prob / prob.sum()
    prob, indices = prob.sort(descending=True)
    if bit_index > len(bit_stream):
        next_token = indices[int(torch.multinomial(prob, 1))].view([1])
    elif bit_index + k > len(bit_stream):
        bit_embed = [int(_) for _ in bit_stream[bit_index:]]
        int_embed = bits2int(bit_embed)
        next_token = indices[int_embed].view([1])
    else:
        bit_embed = [int(_) for _ in bit_stream[bit_index: bit_index + k]]
        int_embed = bits2int(bit_embed)
        next_token = indices[int_embed].view([1])
    return next_token

# def top_k_sample_extract(logits, bit_stream, bit_index, k):
#     prob = torch.exp(logits)
#     prob = prob / prob.sum()
#     prob, indices = prob.sort(descending=True)
#     if bit_index > len(bit_stream):
#         next_token = indices[int(torch.multinomial(prob, 1))].view([1])
#     elif bit_index + k > len(bit_stream):
#         bit_embed = [int(_) for _ in bit_stream[bit_index:]]
#         int_embed = bits2int(bit_embed)
#         next_token = indices[int_embed].view([1])
#     else:
#         bit_embed = [int(_) for _ in bit_stream[bit_index: bit_index + k]]
#         int_embed = bits2int(bit_embed)
#         next_token = indices[int_embed].view([1])
#     return next_token

def do_generate(hidden_bits, bit_index, length, n_ctx, tokenizer, model, device, context, k):
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
            next_token = top_k_sample(next_token_logits, hidden_bits, bit_index, k)
            generated = torch.cat((generated, next_token.unsqueeze(0)), dim=1)
    return generated

def generater(title: str, article_num: int, hidden_bits: str, k, length = 256):
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
            context = context_tokens, k = k)
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

# def extract(text, length, title):
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     bit_index = 0
#     tokenizer = tokenization_bert.BertTokenizer(vocab_file='cache/vocab.txt')
#     model = GPT2LMHeadModel.from_pretrained('model/')
#     model.to(device)
#     model.eval()
#     n_ctx = model.config.n_ctx
#     text = text.replace('\n', '[SEP]')    
#     # text = text.replace('\n', '[CLS]')
#     context = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(title))
#     context = torch.tensor(context, dtype=torch.long, device=device)
#     context = context.unsqueeze(0)
#     text = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))
#     text = torch.tensor(text, dtype=torch.long, device=device)
#     with open('./debug_extract.txt', mode='w') as f:
#         text_list = text.tolist()
#         f.write(str(text_list))
#     # text = text.unsqueeze(0)
#     text_index = len(title)
#     bit_stream = []
#     while bit_index < length:
#         inputs = {'input_ids': context[0][-(n_ctx - 1):].unsqueeze(0)}
#         outputs = model(
#             **inputs)  # Note: we could also use 'past' with GPT-2/Transfo-XL/XLNet (cached hidden-states)
#         next_token_logits = outputs[0][0, -1, :]
#         next_token_logits[tokenizer.convert_tokens_to_ids('[UNK]')] = -float('Inf')
#         next_token_logits[tokenizer.convert_tokens_to_ids('[CLS]')] = -float('Inf')
#         next_token_logits[tokenizer.convert_tokens_to_ids('##')] = -float('Inf')
#         bit_extract = packet_sampling_for_extract(next_token_logits, text[text_index], device)
#         next_token = text[text_index].view([1])
#         text_index += 1
#         context = torch.cat((context, next_token.unsqueeze(0)), dim=1)
#         bit_index += len(bit_extract)
#         bit_stream.append(bit_extract)
#     return "".join(bit_stream)

stream = 'CrazyThursdayVme50'
title = '我很想念你，但是'
stream_b = "".join([format(ord(i), '08b') for i in stream])
text = generater(title, 1, stream_b, 3)
print(text)
stream_extract = extract(text[0], len(stream_b), title)
print(bin2str(stream_extract))
