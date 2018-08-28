from sklearn import svm
import numpy as np
import pylab as pl


def loadData(filename):
    dataMat = []
    labelMat = []
    with open(filename) as f:
        for line in f.readlines():
            # import pdb;pdb.set_trace()
            dataMat.append(list(map(float, line.split()))[0:-1])
            labelMat.append(list(map(float, line.split()))[-1])
        return dataMat, labelMat


if __name__ == "__main__":
    x,y = loadData("data/train.txt")
    X,Y = loadData("data/test.txt")

    # 使用线性方法分类
    clf = svm.SVR(kernel='linear')
    clf.fit(x,y)
    res = clf.predict(X)
    print("kernel is linear", np.corrcoef(Y,res,rowvar=0)[0,1])

    # 使用经向机方法分类
    clf = svm.SVR(kernel='rbf')
    clf.fit(x,y)
    res = clf.predict(X)
    print("kernel is rbf", np.corrcoef(Y,res,rowvar=0)[0,1])

    # 使用线性最小二乘法进行分类
    x = np.mat(x)
    X = np.mat(X)
    y = np.mat(y).T
    Y = np.mat(Y).T
    temp = x.T*x
    ws = temp.I*(x.T*y)
    yPre = X*ws
    print("linear", np.corrcoef(yPre,Y,rowvar=0)[0,1])

    # svr支持向量回归
    x = np.r_[np.random.randn(20,2)-[2,2], np.random.randn(20,2)+[2,2]]
    y = [0]*20 + [1]*20
    clf = svm.SVC(kernel="linear")
    clf.fit(x,y)
    w = clf.coef_[0]
    a = -w[0] /w[1]
    xx = np.arange(-4,4)
    yy = a*xx - clf.intercept_[0] /w[1]
    b = clf.support_vectors_[0]
    yy1 = a*xx + (b[1]-a*b[0])
    b= clf.support_vectors_[-1]
    yy2 = a*xx +b[1] -a*b[0]
    pl.plot(xx,yy,'k-', color='red')
    pl.plot(xx,yy1,'k--', color='green')
    pl.plot(xx,yy2,'k--', color='green')
    pl.plot(clf.support_vectors_[:,0],clf.support_vectors_[:,1], '*', color='red')
    pl.scatter(x[:,0], x[:,1], marker='.', s=60)

    pl.show()

