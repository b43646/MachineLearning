# -*- coding: UTF-8 -*-


from sklearn import linear_model


'''
逻辑回归算法示例:
   使用逻辑回归算法预测用户是否愿意购车

数据格式说明:   
x ：[年龄，年薪]
Y : [购买意愿]

逻辑回归算法基本思路:
- 逻辑回归模型是一个条件概率分布
- 要使用这个模型，需要通过极大似然估计来估算模型参数
- 对似然函数求导，并运用梯度下降法，可以求得极大似然估计值，即模型参数
- 上一步的求解过程需要通过训练样本来进行，最终得出模型参数
'''


x = [[20, 3],
     [23, 7],
     [31, 10],
     [42, 13],
     [50, 7],
     [60, 5]]

y = [0, 1, 1, 1, 0, 0]

lr = linear_model.LogisticRegression()
lr.fit(x, y)

testx = [[28, 8]]
label = lr.predict(testx)
print("predicted Label", label)

prob = lr.predict_proba(testx)
print("probability = ", prob)
