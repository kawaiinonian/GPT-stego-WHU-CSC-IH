from colorama import init

init(autoreset=True)


class valid_bit:  # 有效位
    def __init__(self, b, i):
        self.num = i  # 序号
        self.bit = int(b)  # 数值1
        self.link = []  # 组成成分，7 = 4 + 2 + 1


class check_bit:  # 校验位
    def __init__(self, i):
        self.num = i  # 序号
        self.bit = None  # 数值
        self.link = []  # 校验位


def smallest_check_number(k):
    r = 1
    while 2 ** r - r - 1 < k:
        r += 1  # 得到最小检测位数
    return r


def xor1(lista):
    j = lista[0]
    for i in range(1, len(lista)):
        j = j ^ lista[i]
    return j

def is_standard(string):
    return string.count('1') + string.count('0') == len(string)


def hamming_encode(string):
    checkList.clear()
    hammingList = []
    hammingList.append(0)  # 填补0位，index即下标
    for i in range(1, len(string) + 1):
        locals()['b' + str(i)] = valid_bit(int(string[i - 1]), i)
        hammingList.append(locals()['b' + str(i)])  # 先加入b
    r = smallest_check_number(len(string))
    for j in range(1, r + 1):
        locals()['P' + str(j)] = check_bit(j)
        hammingList.insert(2 ** (j - 1), locals()['P' + str(j)])
        checkList.append(2 ** (j - 1))  # 再插入P
    for i in range(1, len(hammingList)):  # i是有效位，j是检测位
        if i in checkList:
            continue  # 跳过P
        remain = i
        for j in range(len(checkList) - 1, -1, -1):
            if remain >= checkList[j]:
                remain -= checkList[j]
                hammingList[i].link.append(checkList[j])
            if remain == 0:
                break
        for j in hammingList[i].link:
            hammingList[j].link.append(i)  # P的link中加入b的位号
    # 计算检测码
    for j in checkList:
        xor = 0
        for i in hammingList[j].link:
            xor = xor ^ hammingList[i].bit
        hammingList[j].bit = xor
    hamming1 = []
    for i in range(1, len(hammingList)):
        # if i in checkList:  # 检测码
        hamming1.append(str(hammingList[i].bit))
    s = ''.join(hamming1)
    return s


def hamming_decode(string):
    len_a = len(string)
    k = 0

    # 确定k的值
    for i in range(1, len_a):
        tmp = pow(2, i)
        if tmp > len_a:
            k = i
            break

    # 循环创建校验位变量和校验位对应的数据位列表
    for i in range(k):
        exec('P' + str(i) + '=' + str(string[pow(2, i) - 1]))
        exec('D' + str(i) + '=[]')

    # 循环创建数据位变量并确定是否与校验位有关
    for i in range(len_a):
        if bin(i + 1).count('1') == 1:
            continue
        else:
            tmp1 = bin(i + 1)[-1:1:-1]
            for j in range(len(tmp1)):
                if tmp1[j] == '1':
                    exec('D' + str(j) + '.append(' + str(string[i]) + ')')
    K = ''
    tmp2 = ' '
    tmp3 = ' '
    tmp5 = ' '
    for i in range(k):
        exec('tmp2 = D' + str(i))
        exec('tmp3 = P' + str(i))
        tmp4 = xor1(tmp2)
        if tmp4 == tmp3:
            K += '0'
        else:
            K += '1'
    tmp5 = K[::-1]

    a = list(string)
    if tmp5.find('1') != -1:
        int1 = int(tmp5, 2) - 1
        a[int1] = str(int(a[int1]) ^ 1)

    b = ''
    c = [pow(2, i) for i in range(k)]
    for i in range(len_a):
        if i + 1 not in c:
            b += a[i]
    return b


checkList = []
# stream = 'CrazyThursdayVm'
# ham = "".join([format(ord(i), '08b') for i in stream])
# print(ham)

# print(hamming_encode(ham))  # 编码
# print(hamming_decode(hamming_encode(ham)))  # 解码
