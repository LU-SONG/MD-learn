### 宋璐-第1天-计算材料学简介
#### 一、学习笔记
##### 1.1 计算材料学
**计算**方法与分子模拟、虚拟实验，已经继实验、理论之后，成为第三个重要的科学方法，发表学术论文不可或缺。计算需要具备**复制性和再现性**。融合凝聚态物理、材料物理化学、材料力学、计算机编程等多学科知识。
##### 1.2 尺度划分
![尺度](https://github.com/LU-SONG/MD-learn/blob/main/%E8%AE%A1%E7%AE%97%E6%9D%90%E6%96%99%E5%AD%A6_%E7%BB%84%E9%98%9F%E5%85%B1%E8%AF%BB%E7%AC%94%E8%AE%B0/notebook/images/markdown-image-1.png?raw=true)

|   尺度   |   粒子，体系   |     速度，精度     |            理论，方法            |        软件        |                             应用                             |
| :------: | :------------: | :----------------: | :------------------------------: | :----------------: | :----------------------------------------------------------: |
| 纳观尺度 | 电子，一百原子 |      一般，高      |  量子力学，DFT、第一性原理计算   | vasp，gaussian，QE | 分子、晶体结构；表面性质；力学性质；轨道、能带、价态；磁性、SOC；吸收光谱；动力学扩散 |
| 微观尺度 | 原子，上千原子 | 快，较高（势函数） |  牛顿力学，蒙特卡罗、分子动力学  |       lammps       |     材料形变机制；相变；吸附，扩散，缺陷运动，分子自组装     |
| 介观尺度 |     粗粒子     |      慢，一般      |      牛顿力学、统计热力学等      |         —          |                        相关理论不成熟                        |
| 宏观尺度 |     连续体     |   速度与精度相关   | 理论、流体力学，有限元、有限差分 |   ANSYS，ABAQUS    |                    工业应用仿真，宏观模拟                    |

##### 1.3 计算材料学研究方法

建模，方法选择，分析结果，理论解释

#### 二、课后巩固
##### 2.1 不同尺度的物理过程如何相互作用？

纳观和微观尺度：电子、中子、质子组成原子，电子能量是量子化的，以电子云的形式概率分布在原子中。化学反应中，电子转移导致化学键的形成和断裂。

介观和微观尺度：靠近微观的介观尺度上，量子力学和经典力学同时起作用，扫描隧道显微镜(STM)、原子力显微镜(AFM)等都可以观察该两种尺度的现象。

宏观和介观尺度：在靠近宏观的介观尺度上，一般用经典力学来解释两种尺度的现象，例如相变和临界现象。

##### 2.2 精确多尺度模拟方面遇到的主要技术难题是什么？
算力限制：跨尺度模拟需要兼顾两种尺度的特点，目前精确模拟单尺度算力可能仍是一个不充足的状态，原尺度算力可能不足以满足精确跨尺度模拟的需要。目前我们只能计算几百或上千（或上万）个原子，但我们模拟真实体系，可能需要对上亿的原子进行计算；另外还有时间尺度，分子动力学可以做到皮秒或者纳秒，但是当我们真正要解决一个工艺问题时，则需要秒甚至小时级别的模拟。
模型的准确性和可靠性：由于多尺度模拟涉及到多个尺度的相互作用和影响，如何考虑不同因素的影响而构建准确可靠的模型是一个重要难题。

##### 2.3 机器学习如何辅助材料科学研究，特别是在新材料的设计和发现方面，以加速材料性质的预测和优化过程？

目前我所了解的机器学习与材料研究结合手段主要有

高通量计算和实验数据分析，基于数据库，利用机器学习在海量实验数据中发现分子结构、属性之间的相互关系，并找出映射，预测材料性质等。

机器学习可以开发力场，做分子动力学模拟。使用中小体系计算密度泛函理论时，会产生大量的数据，而我们只需要知道原子结构、映射出能量和原子受力，就可以利用机器学习不断迭代、反复学习映射，得到机器学习力场。材料领域涉及的大部分情况下没有力场，或者已有的力场精度不够。引入机器学习可以使得分子动力学力场进一步发展。

##### 2.4 NASA材料基因工程2040规划？

![nasa](https://github.com/LU-SONG/MD-learn/blob/main/%E8%AE%A1%E7%AE%97%E6%9D%90%E6%96%99%E5%AD%A6_%E7%BB%84%E9%98%9F%E5%85%B1%E8%AF%BB%E7%AC%94%E8%AE%B0/notebook/images/image-20231210185648128.png?raw=true)

目标：打通材料到制造体系全链条模型和计算技术，实现利用材料计算化学驱动航天器部组件先进制造技术发展的总体目标

9大要素：计算模型和理论方法；多尺度测试表征工具和方法；优化和优化方法；决策与不确定度量化及管理；验证与确认；数据信息与可视化；工作流程和组织框架；教育培训；计算基础设施。

##### 2.5 我国材料基因组工程的开展情况？

十三五规划中我国材料基因组工程的主要研究任务部分

![我国](https://github.com/LU-SONG/MD-learn/blob/main/%E8%AE%A1%E7%AE%97%E6%9D%90%E6%96%99%E5%AD%A6_%E7%BB%84%E9%98%9F%E5%85%B1%E8%AF%BB%E7%AC%94%E8%AE%B0/notebook/images/image-20231210184825473.png?raw=true)

以及在各领域取得的较多的研究成果

##### 2.6 比较分子动力学方法与蒙特卡罗方法的异同点。

分子动力学和蒙特卡罗方法共同点：两者都使用采样方法，基于统计思想，通过模拟大量粒子的行为来计算系统的宏观性质。

|   区别   |                       MD                       |                        MC                        |
| :------: | :--------------------------------------------: | :----------------------------------------------: |
| 理论基础 | 牛顿力学，求解牛顿方程，模拟粒子在势场中的运动 | 统计力学，基于随机采样统计规律研究体系热平衡状态 |
| 时间演化 |               时间轴，皮秒到纳秒               |          传统MC无时间维度，描述状态跳跃          |
| 温度控制 |            通过速度（动能）控制温度            |      通过Metropolis准则进行状态的接受或拒绝      |

### 宋璐-第2天-Python与科学计算
#### Python列表常用函数&方法

`len(list)`    列表元素个数
`max(list)`   返回列表元素最大值
`min(list)`   返回列表元素最小值
`list(seq)`   将元组转换为列表
`list.append(obj)`  在列表末尾添加新的对象
`list.count(obj)`   统计某个元素在列表中出现的次数
`list.index(obj)`  从列表中找出某个值第一个匹配项的索引位置
`list.reverse()`     反向列表中元素

**数据类型**：**数值**：整型、浮点型、布尔、复数。**None**。**字符串**

**序列**：列表list、元组tuple、集合set、字典dictionary。

**运算符**

**流程控制**：if-elif、while、for、break, contine, pass

`range`序列：`range(1, 100, 2)`表示从1到100，间隔2

help(xxxx)：可查看该函数的用法

**函数**：提高代码的重复利用率

[内置函数]([内置函数 — Python 3.12.0 文档](https://docs.python.org/zh-cn/3/library/functions.html))

函数进阶

[任意实参列表](https://docs.python.org/zh-cn/3/tutorial/controlflow.html#arbitrary-argument-lists)
### 2.2 模块
模块好比工具包

导入模块

![image-20231018163554661](./images/image-20231018163554661.png)

#### numpy

numpy是matlab的平替，支持大量的维度数组和矩阵运算

1000 x 1000 matrix 运算 远远快于 python 

创建数组

```
r= np.random.randint(0, 101, size=(5, 5))	# 创建5x5的随机
```

##### numpy索引

一维数组访问

![image-20231018165319450](./images/image-20231018165319450.png)

多维数组访问

![image-20231018165619482](./images/image-20231018165619482.png)

list的全部索引方式，对于numpy向量、矩阵均可用

数值索引

![image-20231018165906713](./images/image-20231018165906713.png)

布尔索引

![image-20231018165940485](./images/image-20231018165940485.png)

矩阵的转置和重塑

`data.T`转置

`data.reshape`更改维度
##### numpy计算

基本运算

![image-20231018170504979](./images/image-20231018170504979.png)

函数运算

![image-20231018170546946](./images/image-20231018170546946.png)

![image-20231018170559942](./images/image-20231018170559942.png)

向量聚合

![image-20231018170634235](./images/image-20231018170634235.png)

![image-20231018170945260](./images/image-20231018170945260.png)

沿着某行或者某列的聚合，加入axis

![image-20231018171141092](./images/image-20231018171141092.png)

数组运算和广播机制

不同维度数组运算

![image-20231018171314659](./images/image-20231018171314659.png)

数组点乘

1x1+2x100+3x10000=30201

![image-20231018171412529](./images/image-20231018171412529.png)

矩阵相乘

2x3和3x2的矩阵相乘为2x2的矩阵

![image-20231018171711166](./images/image-20231018171711166.png)

##### numpy线性代数

![image-20231018172009215](./images/image-20231018172009215.png)

矩阵求逆

![image-20231018172029210](./images/image-20231018172029210.png)

特征值

![image-20231018172054401](./images/image-20231018172054401.png)

线性方程组

![image-20231018172221748](./images/image-20231018172221748.png)

#### Matplotlib

##### 曲线图

```python
import matplotlib.pyplot as plt
import numpy as np
plt.figure(figsize=(10,8))
X = np.linspace(0, 2*np.pi, 50)
Y = np.sin(X)
```

![image-20231018193908691](./images/image-20231018193908691.png)

```py
plt.title("sin(X) & 2 sin(X)")
plt.xlim((0, np.pi*2))
plt.ylim((-3, 3))
plt.xlabel('X')X
plt.ylabel('Y')
plt.xticks((0, np.pi*0.5, np.pi, np.pi*1.5, np.pi*2)) # 给出刻度坐标
plt.plot(X, Y, label="sin(x)")
plt.plot(X, Y*2, label="2sin(x)")
plt.legend(loc="best")# 显示label
plt.show()
```

##### 柱状图

```py
import matplotlib.pyplot as plt
x = [1,2,3,4,5]
y = [3,6,1,8,2]
plt.bar(x,y,
width=0.3, # 柱子粗细
color='r', # 柱子颜色
alpha=0.3) # 不透明度，值越小越透明
plt.xticks(x,['a','b','c','d','e'])
plt.yticks(y)
plt.show()
```

##### 散点图

```py
import numpy as np
import matplotlib.pyplot as plt
x = np.linspace(0, 2*np.pi, 50)
y = np.sin(x)
plt.scatter(x, y,
color = ‘r’, #点的填充颜色
marker = ‘o’, #散点的形状
edgecolor = ‘k’) #点的边缘颜色
plt.show()
```

```py
import matplotlib.pyplot as plt
import numpy as np
# 生成500个坐标随机的数据点
k = 500
x = np.random.rand(k)
y = np.random.rand(k)
# 随机生成每个点的大小
size = 50 * np.random.rand(k)
# 根据各点坐标(x,y)对应的反正切分配颜色
colour = np.arctan2(x, y)
# 画图并添加颜色栏（colorbar）
plt.scatter(x, y, s=size, c=colour)
# 颜色分布区间
plt.colorbar()
plt.show()
```

分形画图

```py
import numpy as np
import matplotlib.pyplot as plt
def mandelbrot( h,w, maxit=20 ):
    y,x = np.ogrid[ -1.4:1.4:h*1j, -2:0.8:w*1j ]
    c = x+y*1j
    z = c
    divtime = maxit + np.zeros(z.shape, dtype=int)
    for i in range(maxit):
        z = z**2 + c
        diverge = z*np.conj(z) > 2**2
        div_now = diverge & (divtime==maxit)
        divtime[div_now] = i
        z[diverge] = 2
    return divtime
plt.imshow(mandelbrot(400,400))
plt.show()
```

![hh](https://github.com/LU-SONG/MD-learn/blob/e1a795963facbb639f0e1d7ee8c69ee1ef6ee3ab/%E8%AE%A1%E7%AE%97%E6%9D%90%E6%96%99%E5%AD%A6_%E7%BB%84%E9%98%9F%E5%85%B1%E8%AF%BB%E7%AC%94%E8%AE%B0/notebook/images/markdown-image.png)