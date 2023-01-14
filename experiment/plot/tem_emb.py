import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator


plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False
plt.figure(figsize=(15, 10), dpi=100)
temperature_list = [1.0, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50]
emb_150_256 = [75.2, 70.6, 55.0, 56.0, 47.8, 46.6, 37.8, 33.6, 31.2, 27.8, 19.4]
emb_200_256 = [145.2, 111.0, 107.2, 82.4, 74.8, 63.4, 55.0, 49.2, 32.0, 31.8, 23.2]
emb_250_256 = [134.0, 127.4, 105.4, 86.6, 83.0, 76.4, 65.4, 58.0, 46.8, 43.0, 34.0]
plt.plot(temperature_list, emb_150_256, c='red', label="150_256")
plt.plot(temperature_list, emb_200_256, c='blue', label="200_256")
plt.plot(temperature_list, emb_250_256, c='green', label="250_256")

plt.scatter(temperature_list, emb_150_256, c='red')
plt.scatter(temperature_list, emb_200_256, c='blue')
plt.scatter(temperature_list, emb_250_256, c='green')

plt.grid(True, linestyle='--', alpha=0.5)
plt.yticks(range(0, 150, 5))
x_major_locator = MultipleLocator(0.05)
plt.gca().xaxis.set_major_locator(x_major_locator)
plt.xlim(0.49,1.01)
plt.legend(loc='best')


plt.xlabel("temperature", fontdict={'size': 16})
plt.ylabel("完成嵌入时使用的字数", fontdict={'size': 16})
plt.title("temperature对嵌入的影响", fontdict={'size': 20})
plt.show()
