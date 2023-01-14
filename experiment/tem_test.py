import generate_text
import generate_baseline
import generater_ADG
import generater_temperature
import evaluate
import random
from generater_temperature import using_cover

# stream = 'There is an example of information steganography'
# stream_b = "".join([format(ord(i), '08b') for i in stream])
# print(len(stream_b))

stream_b = ""
stream_b_len = 300
for i in range(stream_b_len):
    if random.random() <= 0.5:
        stream_b += "1"
    else:
        stream_b += "0"

length = 256
title = '雨一直下，风一直刮'
batch_num = 5
# temperature_list = [1.0, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50]
temperature_list = [0.55, 0.50]


for j in range(len(temperature_list)):
    print("temperature = " + str(temperature_list[j]))
    tem_acc = []
    text_ppl = []
    tem_ppl = []
    tem_kl = []
    with open("resualt/tem_" + str(stream_b_len) + "_" + str(length) + ".txt", mode='a') as f:
        f.write("temperature = " + str(temperature_list[j]) + "\n")
        for i in range(batch_num):
            print("batch" + str(i) + ":")
            # text generate
            text = generate_text.generater(title, 1, length, i)
            text_temperature = generater_temperature.generater(title, 1, stream_b, length, i, temperature_list[j])

            # extract
            stream_ext_temperature = generater_temperature.extract(text_temperature, len(stream_b), title, temperature_list[j])

            print(text_temperature)
            f.write("batch" + str(i) + ":\n")
            f.write(text + "\n\n")
            f.write(text_temperature + "\n\n")

            f.write("extract_text:\n")
            tem_acc.append(evaluate.stream_evaluate(stream_ext_temperature, stream_b))
            f.write("tem_acc:" + str(tem_acc[i]) + "\n")
            print(tem_acc[i])
            # f.write(evaluate.bin2str(stream_ext_temperature)+ "\n")

            f.write("\nppl:\n")
            text_ppl.append(evaluate.get_ppl(text))
            f.write("text_ppl:" + str(text_ppl[i]) + "\n")

            tem_ppl.append(evaluate.get_ppl(text_temperature))
            f.write("temperature_ppl:" + str(tem_ppl[i]) + "\n")

            f.write("\nkl:\n")
            tem_kl.append(evaluate.get_kl(text_temperature, text))
            f.write("temperature_kl:" + str(tem_kl[i]) + "\n\n")
        f.write(str(tem_acc))
        f.write("\ntem_acc:" + str(sum(tem_acc) / batch_num) + "\n\n")

        f.write(str(text_ppl))
        f.write("\ntext_ppl:" + str(sum(text_ppl) / batch_num) + "\n\n")
        f.write(str(tem_ppl))
        f.write("\ntem_ppl:" + str(sum(tem_ppl) / batch_num) + "\n\n")

        f.write(str(tem_kl))
        f.write("\ntem_kl:" + str(sum(tem_kl) / batch_num) + "\n\n")
        f.write("Embedding rate:" + str(sum(using_cover)/batch_num) + "\n\n")
        using_cover.clear()
        f.close()