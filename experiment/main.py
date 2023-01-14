import generate_text
import generate_baseline
import generater_ADG
import generater_temperature
import evaluate
import random

# stream = 'There is an example of information steganography'
# stream_b = "".join([format(ord(i), '08b') for i in stream])
# print(len(stream_b))

stream_b = ""
stream_b_len = 200
for i in range(stream_b_len):
    if random.random() <= 0.5:
        stream_b += "1"
    else:
        stream_b += "0"

length = 256
title = '雨一直下，风一直刮'
batch_num = 20
temperature = 0.85

ADG_acc = []
tem_acc = []
text_ppl = []
base_ppl = []
ADG_ppl = []
tem_ppl = []
base_kl = []
ADG_kl = []
tem_kl = []

for i in range(batch_num):
    print("batch" + str(i) + ":")
    text = generate_text.generater(title, 1, length, i)
    text_base = generate_baseline.generater(title, 1, stream_b, 3, length, i)
    text_ADG = generater_ADG.generater(title, 1, stream_b, length, i)
    text_temperature = generater_temperature.generater(title, 1, stream_b, length, i, temperature)

    # extract
    stream_ext_ADG = generater_ADG.extract(text_ADG, len(stream_b), title)
    stream_ext_temperature = generater_temperature.extract(text_temperature, len(stream_b), title, temperature)

    with open("resualt/0.85_" + str(stream_b_len) + "_" + str(length) + ".txt", mode='a') as f:
        f.write("batch" + str(i) + ":\n")
        f.write(text + "\n\n")
        f.write(text_base + "\n\n")
        f.write(text_ADG + "\n\n")
        f.write(text_temperature + "\n\n")

        f.write("extract_text:\n")
        ADG_acc.append(evaluate.stream_evaluate(stream_ext_ADG, stream_b))
        f.write("ADG_acc:" + str(ADG_acc[i]) + "\n")
        # f.write(evaluate.bin2str(stream_ext_ADG)+ "\n")
        tem_acc.append(evaluate.stream_evaluate(stream_ext_temperature, stream_b))
        f.write("tem_acc:" + str(tem_acc[i]) + "\n")
        # f.write(evaluate.bin2str(stream_ext_temperature)+ "\n")

        f.write("\nppl:\n")
        text_ppl.append(evaluate.get_ppl(text))
        f.write("text_ppl:" + str(text_ppl[i]) + "\n")

        base_ppl.append(evaluate.get_ppl(text_base))
        print(text_base)
        print(base_ppl)
        f.write("baseline_ppl:" + str(base_ppl[i]) + "\n")

        ADG_ppl.append(evaluate.get_ppl(text_ADG))
        f.write("ADG_ppl:" + str(ADG_ppl[i]) + "\n")

        tem_ppl.append(evaluate.get_ppl(text_temperature))
        f.write("temperature_ppl:" + str(tem_ppl[i]) + "\n")

        f.write("\nkl:\n")
        base_kl.append(evaluate.get_kl(text_base, text))
        f.write("baseline_kl:" + str(base_kl[i]) + "\n")

        ADG_kl.append(evaluate.get_kl(text_ADG, text))
        f.write("ADG_kl:" + str(ADG_kl[i]) + "\n")

        tem_kl.append(evaluate.get_kl(text_temperature, text))
        f.write("temperature_kl:" + str(tem_kl[i]) + "\n\n")

        f.close()

with open("resualt/0.85_" + str(stream_b_len) + "_" + str(length) + ".txt", mode='a') as f:
    f.write(str(ADG_acc))
    f.write("\nADG_acc:" + str(sum(ADG_acc)/batch_num) + "\n\n")
    f.write(str(tem_acc))
    f.write("\ntem_acc:" + str(sum(tem_acc) / batch_num) + "\n\n")

    f.write(str(text_ppl))
    f.write("\ntext_ppl:" + str(sum(text_ppl) / batch_num) + "\n\n")
    f.write(str(base_ppl))
    f.write("\nbase_ppl:" + str(sum(base_ppl) / batch_num) + "\n\n")
    f.write(str(ADG_ppl))
    f.write("\nADG_ppl:" + str(sum(ADG_ppl) / batch_num) + "\n\n")
    f.write(str(tem_ppl))
    f.write("\ntem_ppl:" + str(sum(tem_ppl) / batch_num) + "\n\n")

    f.write(str(base_kl))
    f.write("\nbase_kl:" + str(sum(base_kl) / batch_num) + "\n\n")
    f.write(str(ADG_kl))
    f.write("\nADG_kl:" + str(sum(ADG_kl) / batch_num) + "\n\n")
    f.write(str(tem_kl))
    f.write("\ntem_kl:" + str(sum(tem_kl) / batch_num) + "\n")
    f.close()