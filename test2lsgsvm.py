#from test2lsg.py
# now look at similar results, but for support vector machines with regression!
import pandas as pd
import numpy as np
import scipy
import matplotlib
from datetime import datetime,timedelta
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from sklearn.svm import SVR

CUSTOMERDATA=r"c:\users\klar\documents\EKPC\NCEI\data"
T2MFN_ORIG="%s\orig0707T2wna.txt" % (CUSTOMERDATA)
T2MFN_FILLED="%s\\filled0707T2.txt" % (CUSTOMERDATA)

#CUSTOMERDATA="/home/kristin/test/ekpcdata"
#T2MFN_ORIG="%s/orig0707T2wna.txt" % (CUSTOMERDATA)
#T2MFN_FILLED="%s/filled0707T2.txt" % (CUSTOMERDATA)

class Target(object):
    def __init__(self,fn=T2MFN_FILLED):
        todatetime=lambda x:datetime.strptime(str(x),"%Y%m%d%H")
        self.todatetime=todatetime
        self.fn=fn
        print("Reading from",fn)
        self.df=pd.read_csv(fn,converters={'Timestamp':todatetime},index_col='Timestamp')
        self.df.drop('row',axis=1, inplace=True)
        self.means=self.df.mean().values
        self.stds=self.df.std().values
        self.ntimes,self.ncols=self.df.shape
        self.ndays=self.ntimes/24
        
        
#%matplotlib inline
#from target import Target
def main():
    target=Target()
    target.df.plot()

    print(target.means)

    print(target.stds)

    print(target.ntimes)

    print(target.ndays)
    
    print(target.df.shape)
    print(target.df.describe())




# try my original, not flattened, with scaling
    xscaler=StandardScaler().fit(target.df[0:8760])
    print(xscaler.mean_)
    print(xscaler.scale_)
    yscaler=StandardScaler().fit(target.df.ix[48:8808,['T2m_KLEX']])
    print(yscaler.mean_)
    print(yscaler.scale_)
    X_train=xscaler.transform(target.df[0:8760])
    X_test=xscaler.transform(target.df[8760:17496])
    Y_train=yscaler.transform(target.df.ix[48:8808,['T2m_KLEX']])
    Y_test=yscaler.transform(target.df.ix[8808:17544, ['T2m_KLEX']])
    print(X_train.shape)
    print(X_test.shape)
    print(Y_train.shape)
    print(Y_test.shape)
    regrss = linear_model.LinearRegression()
    regrss.fit(X_train, Y_train)
    svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
    svr_lin = SVR(kernel='linear', C=1e3)
    svr_poly = SVR(kernel='poly', C=1e3, degree=2)
    svr_rbf.fit(X_train, Y_train)
    svr_lin.fit(X_train, Y_train)
    svr_poly.fit(X_train, np.ravel(Y_train))
#    y_rbf = svr_rbf.fit(X_train, Y_train).predict(X_test)
#    y_lin = svr_lin.fit(X_train, Y_train).predict(X_test)
#    y_poly = svr_poly.fit(X_train, Y_train).predict(X_test)
    y_rbf = svr_rbf.predict(X_test)
    y_lin = svr_lin.predict(X_test)
    y_poly = svr_poly.predict(X_test)
    y_ols = regrss.predict(X_test)
#    print(pd.DataFrame(X_train).describe())
#    print(pd.DataFrame(Y_train).describe())
#    print(pd.DataFrame(Y_test).describe())
#    print(pd.DataFrame(X_test).describe())
#    print('Coefficients: \n', regrss.coef_)
#    print('Intercept \n', regrss.intercept_)   
    print("Mean squared error scaled Ordinary Least Squares: ")
    print(np.mean(np.power(y_ols - Y_test,2),axis=0))
    print("Mean squared error Ordinary Least Squares: ")
    print(np.mean(np.power(yscaler.inverse_transform(y_ols) - yscaler.inverse_transform(Y_test),2),axis=0))
    print("Mean squared error scaled RBF: ")
    print(np.mean(np.power(y_rbf - Y_test,2),axis=0))
    print("Mean squared error RBF: ")
    print(np.mean(np.power(yscaler.inverse_transform(y_rbf) - yscaler.inverse_transform(Y_test),2),axis=0))
    print("Mean squared error scaled SVM linear: ")
    print(np.mean(np.power(y_lin - Y_test,2),axis=0))
    print("Mean squared error SVM linear: ")
    print(np.mean(np.power(yscaler.inverse_transform(y_lin) - yscaler.inverse_transform(Y_test),2),axis=0))
    print("Mean squared error scaled SVM polynomial: ")
    print(np.mean(np.power(y_poly - Y_test,2),axis=0))
    print("Mean squared error SVM polynomial: ")
    print(np.mean(np.power(yscaler.inverse_transform(y_poly) - yscaler.inverse_transform(Y_test),2),axis=0))


#    pd.DataFrame(regr.predict(X_test)).plot()
#    pd.DataFrame(regr.predict(X_test)).describe()
#    pd.DataFrame(regr.predict(X_test)-Y_test).plot()
#    pd.DataFrame(regr.predict(X_test)-Y_test).describe()


if __name__ == "__main__":
    main()
