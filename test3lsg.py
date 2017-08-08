#from test2lsgsvm.py
# Try a couple different models and look at errors with cross validation??
#  Try for KLEX only, and try one model for all hours for now.
# test in ipython if they can do multiple target values
import pandas as pd
import numpy as np
import scipy
import matplotlib
from datetime import datetime,timedelta
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from sklearn.svm import SVR
from sklearn.cross_validation import KFold
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet, SGDRegressor, BayesianRidge
from sklearn.kernel_ridge import KernelRidge

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
    print(target.df.corr())
    
    print(target.df.shape)
    print(target.df.describe())
    xscaler=StandardScaler().fit(target.df[0:8760])
    print(xscaler.mean_)
    print(xscaler.scale_)
    yscaler=StandardScaler().fit(target.df.ix[48:8808,['T2m_KLEX']])
    print(yscaler.mean_)
    print(yscaler.scale_)
    X_train=xscaler.transform(target.df[0:8760])
    X_test=xscaler.transform(target.df[8760:17496])
    Y_train=np.ravel(yscaler.transform(target.df.ix[48:8808,['T2m_KLEX']]))
#    Y_test=np.ravel(yscaler.transform(target.df.ix[8808:17544, ['T2m_KLEX']]))
    Y_test=np.ravel(target.df.ix[8808:17544, ['T2m_KLEX']])
    X_total=xscaler.transform(target.df[0:17496])
    Y_total=np.ravel(yscaler.transform(target.df.ix[48:17544,['T2m_KLEX']]))
       
    print(X_train.shape)
    print(X_test.shape)
    print(Y_train.shape)
    print(Y_test.shape)

    a = 0.3
    for name,met in [
            ('linear regression', LinearRegression()),
            ('lasso', Lasso(fit_intercept=True, alpha=a)),
            ('ridge', Ridge(fit_intercept=True, alpha=a)),
            ('elastic-net', ElasticNet(fit_intercept=True, alpha=a)),
            ('SVM linear', SVR(kernel='linear', C=1e3)),
            ('SVM poly', SVR(kernel='poly', C=1e3, degree=2)),
            ('SVM rbf', SVR(kernel='rbf', C=1e3, gamma=0.1 ))
            ('Stochastic Gradient Descent loss OLS', SGDRegressor(penalty='l2', alpha=0.15, n_iter=200,loss="squared_loss"))
            ('Stochastic Gradient Descent loss Huber', SGDRegressor(penalty='l2', alpha=0.15, n_iter=200,loss="huber"))
            ('Stochastic Gradient Descent loss EPS', SGDRegressor(penalty='l2', alpha=0.15, n_iter=200,loss="epsilon_insensitive"))
            ('Kernel Ridge linear', KernelRidge(kernel='linear', alpha=1.0))
            ('Kernel Ridge rbf', KernelRidge(kernel='rbf', alpha=1.0))
            ('Bayesian Ridge', BayesianRidge())
            ]:
        met.fit(X_train,Y_train)
     # p = np.array([met.predict(xi) for xi in x])
        p = yscaler.inverse_transform(met.predict(X_test))
        e = p-Y_test
        total_error = np.dot(e,e)
        rmse_train = np.sqrt(total_error/len(p))

        kf = KFold(len(X_total), n_folds=10)
        err = 0
        for train,test in kf:
            met.fit(X_total[train],Y_total[train])
            p = yscaler.inverse_transform(met.predict(X_total[test]))
            e = p-yscaler.inverse_transform(Y_total[test])
            err += np.dot(e,e)

        rmse_10cv = np.sqrt(err/len(X_total))
        print('Method: %s' %name)
        print('RMSE on training: %.4f' %rmse_train)
        print('RMSE on 10-fold CV: %.4f' %rmse_10cv)
        print("\n")   


# try my original, not flattened, with scaling
#    svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
#    svr_lin = SVR(kernel='linear', C=1e3)
#    svr_poly = SVR(kernel='poly', C=1e3, degree=2)
#    svr_rbf.fit(X_train, Y_train)
#    svr_lin.fit(X_train, Y_train)
#    svr_poly.fit(X_train, np.ravel(Y_train))
#    y_rbf = svr_rbf.fit(X_train, Y_train).predict(X_test)
#    y_lin = svr_lin.fit(X_train, Y_train).predict(X_test)
#    y_poly = svr_poly.fit(X_train, Y_train).predict(X_test)
#    y_rbf = svr_rbf.predict(X_test)
#    y_lin = svr_lin.predict(X_test)
#    y_poly = svr_poly.predict(X_test)
#    y_ols = regrss.predict(X_test)
#    print(pd.DataFrame(X_train).describe())
#    print(pd.DataFrame(Y_train).describe())
#    print(pd.DataFrame(Y_test).describe())
#    print(pd.DataFrame(X_test).describe())
#    print('Coefficients: \n', regrss.coef_)
#    print('Intercept \n', regrss.intercept_)   
#    print("Mean squared error scaled Ordinary Least Squares: ")
#    print(np.mean(np.power(y_ols - Y_test,2),axis=0))
#    print("Mean squared error Ordinary Least Squares: ")
#    print(np.mean(np.power(yscaler.inverse_transform(y_ols) - yscaler.inverse_transform(Y_test),2),axis=0))


#    pd.DataFrame(regr.predict(X_test)).plot()
#    pd.DataFrame(regr.predict(X_test)).describe()
#    pd.DataFrame(regr.predict(X_test)-Y_test).plot()
#    pd.DataFrame(regr.predict(X_test)-Y_test).describe()


if __name__ == "__main__":
    main()
# output
Method: linear regression
RMSE on training: 0.5519
RMSE on 10-fold CV: 0.5501


Method: lasso
RMSE on training: 0.6259
RMSE on 10-fold CV: 0.6660


Method: ridge
RMSE on training: 0.5519
RMSE on 10-fold CV: 0.5501


Method: elastic-net
RMSE on training: 0.5781
RMSE on 10-fold CV: 0.6009

# scaled output
Method: linear regression
RMSE on training: 5.7877
RMSE on 10-fold CV: 5.7689


Method: lasso
RMSE on training: 6.5632
RMSE on 10-fold CV: 6.9842


Method: ridge
RMSE on training: 5.7875
RMSE on 10-fold CV: 5.7689


Method: elastic-net
RMSE on training: 6.0618
RMSE on 10-fold CV: 6.3016