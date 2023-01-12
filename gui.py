from tkinter import *
import tkinter.messagebox as msgbox
from generater import extract, generater, bin2str
from Hamming import hamming_encode, hamming_decode
# 隐写窗口
win = Tk()
win.title('中文文本生成隐写')
win.geometry('580x320')
win.resizable(0, 0)
canvas=Canvas(win, width=500, height=320)
canvas.pack()
canvas.create_line(250,0,250,320,width=1)
# 文章开头
Label(text='文章开头:').place(x=20, y=80)
input1 = Entry(win)
input1.place(x=100, y=80)
# 生成文章数量
Label(text='生成文章数量:').place(x=20, y=120)
input2 = Entry(win)
input2.place(x=100, y=120)
# 嵌入内容
Label(text='嵌入内容:').place(x=20, y=160)
input3 = Entry(win)
input3.place(x=100, y=160)
# 信息载体
Label(text='信息载体:').place(x=320, y=80)
input4 = Entry(win)
input4.place(x=400, y=80)
# 信息长度
Label(text='信息长度:').place(x=320, y=120)
input5 = Entry(win)
input5.place(x=400, y=120)
# 文章开头
Label(text='文章开头:').place(x=320, y=160)
input6 = Entry(win)
input6.place(x=400, y=160)
# 隐写
def hide():
    input_1 = input1.get()
    input_2 = int(input2.get())
    input_3 = input3.get()
    input_3 = "".join([format(ord(i), '08b') for i in input_3])
    # input_3 = hamming_encode(input_3)
    texts = generater(title=input_1, article_num=input_2, hidden_bits=input_3)
    with open('generate_texts/text.txt', mode='w') as f:
        for text in texts:
            f.write(text)
            f.write('\n\n')
    msgbox.showinfo('隐写成功')
# 提取
def get():
    input_4 = input4.get()
    input_5 = int(input5.get())
    input_6 = input6.get()
    texts = extract(input_4, input_5, input_6)
    # texts = hamming_decode(texts)
    with open('extract/text.txt', mode='w') as f:
        f.write(bin2str(texts))
        f.write('\n\n')
    msgbox.showinfo('提取成功')
Button(text='隐写', command=hide).place(x=120, y=200, width=120)
Button(text='提取', command=get).place(x=420, y=200, width=120)
win.mainloop()