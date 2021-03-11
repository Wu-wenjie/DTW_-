import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

'''
    DTW,Dynamic Time Warping，动态时间规整
'''

# 定义两个信号x，y
x = np.array([1,2,4,6,4,3,1,0])
y = np.array([1,2,3,4,6,4,2,1])

# 画出两个信号的图像，可以看到两个信号是很相似，但是在时间轴上有位移
plt.plot(range(len(x)), x, 'r', label = 'x')
plt.plot(range(len(y)), y, 'b', label = 'y')

plt.legend()
# plt.show()
'''
 DTW的算法是通过衡量两个信号的相似性，找到两个信号对齐的点与点之间的对应关系
 相似性的衡量可以通过对两个信号对应点之间的欧式距离进行衡量
 一句话总结：
 DTW通过衡量信号之间的相似性，得到两个信号点与点之间的映射关系
'''

# 构建信号x和y之间的距离矩阵
# 首先初始化一个零矩阵用作存储信号x，y之间的距离

xydist = np.zeros((len(x),len(y)))
# print(xydist)
# print("\n")

# 然后计算信号x，y各个点之间的距离，并存储在xydist矩阵中
for ii in range(len(x)):
    for jj in range(len(y)):
        xydist[ii, jj] = (x[ii]-y[jj])**2
# print(xydist)

# 下面可视化距离矩阵xydist

def visulize_xydist(xydist):
    fig = plt.imshow(xydist, interpolation='nearest')
    plt.gca().invert_yaxis()
    plt.xlabel('y') # 显示标签
    plt.ylabel('x') # 显示标签
    plt.grid() # 画网格
    plt.colorbar() # 显示色阶的颜色栏
    plt.show()

# visulize_xydist(xydist)

'''
    上面是信号x和y之间的距离矩阵，简单地说，就是距离越近，信号越相似
    如何有效的把所有的距离最小的点找到呢？
    可以构建距离累积矩阵，即一边走路，一边累加距离，得到距离累计矩阵
    最后再遍历这个距离累计矩阵，得到距离最小的路径，即x，y信号各点之间的对应关系
    1.构建距离累计矩阵xy_acc_dist
      构建累计矩阵，需要按照一定规则遍历这个xydist矩阵，得到距离累计矩阵
      这个规则为：
      · 路径必须从（0,0）开始，在（M,N）结束
            -其中M，N是x，y的长度
      · 路径不可以回头，只能向前走
            -即，从点（x，j）向前只能，向右（i+1，j）走，向上（i，j+1）走，向对角线（i+1，j+1）走        
'''

# 首先初始化，这个距离累计矩阵xy_acc_dist
xy_acc_dist = np.zeros((len(x),len(y)))
# 从位置（0,0）开始，且将xy_acc_dist[0,0]设为xydist[0,0]
xy_acc_dist[0,0] = xydist[0,0]
# 画个图看看
# visulize_xydist(xy_acc_dist)

# 再计算第一行的累计距离
for ii in range(1, len(y)):
    xy_acc_dist[0,ii] = xy_acc_dist[0,ii-1] + xydist[0,ii]

# 再计算第一的累计距离
for ii in range(1, len(x)):
    xy_acc_dist[ii,0] = xy_acc_dist[ii-1,0] + xydist[ii,0]

# 对于除了第一行，第一列的元素，采用如下公式计算距离累计矩阵
# xy_acc_dist[i,j] = min{xy_acc_dist[i-1,j-1],xy_acd_dist[i-1,j],xy_acc_dist[i,j-1]} + xydist[i,j]

for ii in range(1,len(x)):
    for jj in range(1,len(y)):
        xy_acc_dist[ii,jj] = min(xy_acc_dist[ii-1,jj], xy_acc_dist[ii-1,jj-1], xy_acc_dist[ii,jj-1]) + xydist[ii,jj]

# 再画个图看看
# visulize_xydist(xy_acc_dist)

'''
    1.寻找距离最短路径
      从最后一个点[M,N]出发，向后不断寻找距离最小的点
      最简单的寻找方法，就是每次只寻找
      · 对角线方向xy_acc_dist[i-1,j-1]
      · 左边xy_acc_dist[i-1,j]
      · 右边xy_acc_dist[i,j-1]
        这三个方向。
'''

# path的起点是时间序列的终点
path1, path2 = [len(x) - 1], [len(y) - 1]
ii, jj = len(x) - 1, len(y) - 1
while ii > 0 and jj > 0:
    if ii == 0:
        # 如果遍历到了x的第一个点，那么剩下的y点都映射到这一点
        jj = jj - 1
    elif jj == 0:
        # 如果遍历到了y的第一个点，那么剩下的x都映射到这一点
        ii = ii - 1
    else:
        direction = np.argmin([xy_acc_dist[ii-1,jj], xy_acc_dist[ii,jj-1], xy_acc_dist[ii-1,jj-1]])
        if direction == 0:
            # 第一个最小
            ii = ii - 1
        elif direction == 1:
            # 第二个最小
            jj = jj - 1
        else:
            # 第三个最小
            ii = ii - 1
            jj = jj - 1
    path1.append(ii)
    path2.append(jj)
# path的结尾就是两条时间序列的起始点
path1.append(0)
path2.append(0)
# 因为是倒着找路径的，再把它正过来
path1 = path1[::-1]
path2 = path2[::-1]
# 打印看看
print("path1\n",path1)
print("path2\n",path2)

# 可视化最短路径

plt.plot(path2,path1)

# 看看效果如何
# 将信号x和信号y之间的对应关系画出来
plt.plot(range(len(x)), x, 'r', label = 'x')
plt.plot(range(len(y)), y, 'b', label = 'y')
plt.legend()
for i in range(len(path1)):
    plt.plot([path1[i],path2[i]],[x[path1[i]],y[path2[i]]],'g')

visulize_xydist(xy_acc_dist)
