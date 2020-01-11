import numpy as np
import matplotlib.pyplot as plt
##########################################
##########################################
##########################################
#原始数据标准化，预处理数据
##########################################
##########################################
########################################
##定义存储输入数据x和目标数据y
x, y = [], []
##遍历数据集
for sample in open("./_Data/prices.txt", "r"):
    ##将逗号作为参数传入
    xx, yy = sample.split(",")
    ##将数据转换为浮点数
    x.append(float(xx))
    y.append(float(yy))
##读取完数据后，将它们转化为Numpy数组以方便进一步处理   
x, y = np.array(x), np.array(y)

##数据标准化残差
x = (x - x.mean()) / x.std()
##将原始数据以散点图展示
plt.figure()
plt.scatter(x, y, c="g", s=10)
plt.show()

##########################################
##########################################
##########################################
#训练数据
##########################################
##########################################
########################################

##在（-2，4之间取100个点作为画图基础）
x0 = np.linspace(-2, 4, 100)

##多项式拟合模型构建
# Get regression model under LSE criterion with degree 'deg'
def get_model(deg):
    return lambda input_x=x0: np.polyval(np.polyfit(x, y, deg), input_x)

##苹果模型构建,损失函数衡量模型
# Get the cost of regression model above under given x, y
def get_cost(deg, input_x, input_y):
    return 0.5 * ((get_model(deg)(input_x) - input_y) ** 2).sum()


##定义测试参数集并根据它进行试验
# Set degrees
test_set = (1, 4, 10)
for d in test_set:
    print(get_cost(d, x, y))

## 绘制相应图像
## Visualize results
plt.scatter(x, y, c="g", s=20)
for d in test_set:
    plt.plot(x0, get_model(d)(), label="degree = {}".format(d))
    
##将横轴，纵轴范围分别限制在（-2，4）（10^5,8x10^5）
plt.xlim(-2, 10)
plt.ylim(1e5, 8e5)
    
    
#图形显示（label）
plt.legend()
plt.show()