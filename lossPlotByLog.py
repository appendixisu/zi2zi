# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def getValue(str):
    a = str.split(':')
    return float(a[1])

def trend_line(x,y):
    (arg1, arg2) = np.polyfit(x,y,1) # 利用 polyfit 幫我們算出資料 一階擬合的 a, b 參數
    p = np.poly1d((arg1, arg2)) # 做出公式, print 的結果是 coefficients[0] * X + coefficients[1]
    #coefficient_of_dermination = r2_score(y, p(x)) // 計算相關係數用，這裡沒有用到
    trend_line = x * arg1 + arg2
    return trend_line

d_loss = []
g_loss = []
category_loss = []
cheat_loss = []
const_loss = []
l1_loss = []
tv_loss = []
with open("shlog.log", "r") as f:  # open your log file
    for line in f:  # read it line by line
        if line.startswith('Sample:'):
            continue
        array = line.split(',')
        for item in array:
            item = item.replace(" ", "")
            if item.startswith('d_loss'):
                value = getValue(item)
                d_loss.append(value)
            elif item.startswith('g_loss'):
                value = getValue(item)
                g_loss.append(value)
            elif item.startswith('category_loss'):
                value = getValue(item)
                category_loss.append(value)
            elif item.startswith('cheat_loss'):
                value = getValue(item)
                cheat_loss.append(value)
            elif item.startswith('const_loss'):
                value = getValue(item)
                const_loss.append(value)
            elif item.startswith('const_loss'):
                value = getValue(item)
                const_loss.append(value)
            elif item.startswith('l1_loss'):
                value = getValue(item)
                l1_loss.append(value)
            elif item.startswith('tv_loss'):
                value = getValue(item)
                tv_loss.append(value)
            
x = np.arange(len(d_loss))

plt.title('d_loss and g_loss')
plt.xlabel("epoch x batches")
plt.ylabel("score")
plt.scatter(x, d_loss, marker='.', label='d_loss')
plt.scatter(x, g_loss, marker='.', label='g_loss')
plt.plot(x, trend_line(x,d_loss), label='trend_d_loss')
plt.plot(x, trend_line(x,g_loss), label='trend_g_loss')

# plt.plot(const_loss, label='const_loss')
gap = int(int(len(d_loss) / 10) / 20) * 20

plt.xticks(np.arange(0, len(d_loss), gap))
plt.legend()
plt.grid(True)
plt.show()



# def mcd(a, b):
#     resto = 0
#     while(b > 0):
#         resto = b
#         b = a % b
#         a = resto
#     return a

# N = 1200

# n = list (range (N))
# an = [1,1]

# for i in range (2,N):
#     k = i-1
#     if mcd (n[i], an[k]) == 1:
#         an.append (n[i] + 1 + an[k])
#     else:
#         an.append (an[k]/mcd (n[i], an[k]))

# fig = plt.figure ()
# ax = fig.add_subplot (111)
# ax.grid (True)
# ax.set_xlim(0, N*1.1)
# ax.set_ylim(min(an), max(an))
# pt, = ax.plot([],[],'ko', markersize=2)

# def init ():
#     pt.set_data([], [])
#     return pt,

# def animate(i):
#     pt.set_data (n[:i], an[:i])
#     return pt,

# ani = FuncAnimation (fig, animate, frames=N, init_func=init, interval=50, blit = True)

# plt.show ()