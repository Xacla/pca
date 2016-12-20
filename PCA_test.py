import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.decomposition

dim=2
data=np.loadtxt("/Users/tomita/Desktop/課題研究/PCA/6_7.csv",delimiter=",")
#x,yの平均を算出し平均_*を中心とした座標に変換
x=data[:,0]
y=data[:,1]
av_x=np.average(x)
av_y=np.average(y)
after_x = x - av_x
after_y = y - av_y


la=np.linalg
pca=sklearn.decomposition.PCA(dim)

#固有値、固有ベクトルを調べる
result=pca.fit_transform(data);
E_xi_eta = pca.get_covariance()
v,u=la.eig(E_xi_eta)



#データ保存
np.savetxt("result6_7.csv", result, delimiter=",")
np.savetxt("6_7kyousan.txt", E_xi_eta, delimiter=",")
np.savetxt("6_7u.csv", u, delimiter=",")#固有ベクトル
np.savetxt("6_7v.csv", v, delimiter=",")#固有値

#xi,etaをプロット
xi=u[0][0]* after_x + u[1][0] * after_y
eta=u[0][1]* after_x + u[1][1] * after_y

plt.scatter(xi,eta,c="g")
filename = '6_7_plot.jpg'
plt.savefig(filename)
