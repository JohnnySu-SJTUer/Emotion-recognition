import numpy as np
import matplotlib.pyplot as plt
 
plt.figure(1) # 创建图表1
plt.figure(2) # 创建图表2
 
x = np.linspace(0, 100, 21)
y_loss = [3.2258, 2.3143, 2.0600, 1.9332, 1.8688, 1.8439, 1.8170, 1.8047, 1.7887, 1.7782, 1.7696, 1.7609, 1.7549, 1.7434, 1.7317, 1.7201, 1.7103, 1.6978, 1.6878, 1.6784, 1.6687]
y_train_acc = [0.1573, 0.2093, 0.2097, 0.2126, 0.2205, 0.2295, 0.2461, 0.2565, 0.2706, 0.2775, 0.2828, 0.2856, 0.2889, 0.2914, 0.2977, 0.3005, 0.3067, 0.3160, 0.3220, 0.3276, 0.3343]
y_val_acc = [0.2771, 0.2771, 0.2793, 0.2789, 0.2783, 0.2774, 0.2870, 0.2991, 0.2981, 0.2972, 0.2873, 0.2895, 0.2774, 0.2774, 0.2771, 0.2771, 0.2771, 0.2771, 0.2771, 0.2771, 0.2771]
plt.figure(1)  #❶ # 选择图表1
plt.plot(x, y_loss)
plt.title('Training loss')
plt.xlabel('Epoch')
plt.figure(2)  #❶ # 选择图表1
plot_train = plt.plot(x, y_train_acc, ".-", label="$train$", color='blue')
plot_val = plt.plot(x, y_val_acc, ".-", label="$val$", color='orange')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.legend()

plt.show()