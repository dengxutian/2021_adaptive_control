# @Time : 2021/8/3 10:06
# @Author : Deng Xutian
# @Email : dengxutian@126.com

def elasticity(x):
    stiffness = 0.5
    return stiffness * x

def pid(x, k, e):
    x_ = x + k * e
    return x_