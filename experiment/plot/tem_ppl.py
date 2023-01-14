import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator

plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False
plt.figure(figsize=(15, 10), dpi=100)
temperature_list = [1.0, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50]
ppl_150_256 = [5.38, 5.57, 5.67, 6.48, 7.22, 8.04, 8.09, 10.06, 11.28, 9.91, 14.35]
ppl_200_256 = [5.91, 5.59, 5.69, 7.51, 6.77, 8.63, 8.80, 9.15, 12.36, 13.66, 14.92]
ppl_250_256 = [6.30, 6.61, 7.78, 7.95, 9.10, 9.81, 11.14, 11.92, 11.83, 14.42, 19.32]
plt.plot(temperature_list, ppl_150_256, c='red', label="150_256")
plt.plot(temperature_list, ppl_200_256, c='blue', label="200_256")
plt.plot(temperature_list, ppl_250_256, c='green', label="250_256")

plt.scatter(temperature_list, ppl_150_256, c='red')
plt.scatter(temperature_list, ppl_200_256, c='blue')
plt.scatter(temperature_list, ppl_250_256, c='green')

plt.grid(True, linestyle='--', alpha=0.5)
plt.yticks(range(0, 22, 1))
x_major_locator = MultipleLocator(0.05)
plt.gca().xaxis.set_major_locator(x_major_locator)
plt.xlim(0.49,1.01)
plt.legend(loc='best')


plt.xlabel("temperature", fontdict={'size': 16})
plt.ylabel("ppl", fontdict={'size': 16})
plt.title("temperature对ppl的影响", fontdict={'size': 20})
plt.show()
