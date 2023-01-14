import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator

plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False
plt.figure(figsize=(15, 10), dpi=100)
temperature_list = [1.0, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50]
kl_150_256 = [0.558, 0.534, 0.530, 0.531, 0.539, 0.520, 0.523, 0.524, 0.509, 0.522, 0.608]
kl_200_256 = [0.512, 0.486, 0.542, 0.509, 0.517, 0.519, 0.535, 0.523, 0.505, 0.484, 0.509]
kl_250_256 = [0.556, 0.503, 0.502, 0.464, 0.554, 0.531, 0.562, 0.508, 0.526, 0.535, 0.544]
plt.plot(temperature_list, kl_150_256, c='red', label="150_256")
plt.plot(temperature_list, kl_200_256, c='blue', label="200_256")
plt.plot(temperature_list, kl_250_256, c='green', label="250_256")

plt.scatter(temperature_list, kl_150_256, c='red')
plt.scatter(temperature_list, kl_200_256, c='blue')
plt.scatter(temperature_list, kl_250_256, c='green')

plt.grid(True, linestyle='--', alpha=0.5)
y_major_locator = MultipleLocator(0.02)
plt.gca().yaxis.set_major_locator(y_major_locator)
x_major_locator = MultipleLocator(0.05)
plt.gca().xaxis.set_major_locator(x_major_locator)
plt.ylim(0.45,0.65)
plt.xlim(0.49,1.01)
plt.legend(loc='best')


plt.xlabel("temperature", fontdict={'size': 16})
plt.ylabel("kl", fontdict={'size': 16})
plt.title("temperature对kl的影响", fontdict={'size': 20})
plt.show()
