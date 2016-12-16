import numpy as np
import pandas as pd
import matplotlib as plt
import sklearn.decomposition

dim=2
data=np.loadtxt("6.7.csv",delimiter=",")
la=np.linalg
pca=sklearn.decomposition.PCA(dim)

result=pca.fit_transform(data);
E = pca.get_covariance()
w,v=la.eig(E)

np.savetxt("result6_7.csv", result, delimiter=",")
np.savetxt("6_7kyousan.txt", E, delimiter=",")
np.savetxt("6_7w.csv", w, delimiter=",")
np.savetxt("6_7v.csv", v, delimiter=",")
print(result);
