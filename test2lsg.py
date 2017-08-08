import pandas as pd
import numpy as np
import scipy
import matplotlib
from datetime import datetime,timedelta
from sklearn.cross_validation import train_test_split,KFold,cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from sklearn import svm
from sklearn import ensemble
from sklearn import feature_selection
import statsmodels
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

   # train on the first year, evaluate on the second year, all 7 sites, 2 day lag
# start training on one temperature  365*24 = 8760
# try all temps to start with, then move on to one model for each hour of the day
    X_train = target.df[0:8760]
    X_test = target.df[8760:17496]
    print(X_train.shape)
    print(X_test.shape)
#    thisy = target.df[['T2m_KLEX']]
#    Y_train = thisy[48:8808]
#    Y_test = thisy[8808:17544]
#    Y_train = target.df.ix[48:8808,['T2m_KLEX','T2m_KSDF']]
#    Y_test = target.df.ix[8808:17544, ['T2m_KLEX','T2m_KSDF']]
    Y_train = target.df.ix[48:8808,:]
    Y_test = target.df.ix[8808:17544, :]
    print(Y_train.shape)
    print(Y_test.shape)
    regr = linear_model.LinearRegression()
    print(X_train.describe())
    print(Y_train.describe())
    regr.fit(X_train, Y_train)
    print('Coefficients: \n', regr.coef_)
    print('Intercept \n', regr.intercept_)   
    print(Y_test.describe())
    print(X_test.describe())
    print("Mean squared error: ")
    print(np.mean(np.power(regr.predict(X_test) - Y_test,2),axis=0))
# Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % regr.score(X_test, Y_test))

    pd.DataFrame(regr.predict(X_test)).plot()
    pd.DataFrame(regr.predict(X_test)).describe()
    pd.DataFrame(regr.predict(X_test)-Y_test).plot()
    pd.DataFrame(regr.predict(X_test)-Y_test).describe()

# try this again with scaling this time!  Is the mean square error the same?
# also try a seperate model for each location and each hour of the day, how does that compare?
#  first temperature for 7 sites,  then hour 1 for all 7 sites,
# So first record does contain all 7 sites with all 24 hours of the first day.  Just the 7 sites are one every 7,  not consecutive,
    bignew = np.array(target.df).reshape(731,168,order='C')
    print(bignew.shape)
    bignew[0:5,:]
    X_train = bignew[0:365,:]
    X_test = bignew[365:729,:]
    print(X_train.shape)
    print(X_test.shape)
    Y_train = bignew[2:367,:]
    Y_test = bignew[367:731, :]
    print(Y_train.shape)
    print(Y_test.shape)
    regr24 = linear_model.LinearRegression()
    print(pd.DataFrame(X_train).describe())
    print(pd.DataFrame(Y_train).describe())
    regr24.fit(X_train, Y_train)
    print('Coefficients: \n', regr24.coef_)
    print('Intercept \n', regr24.intercept_)   
    print(pd.DataFrame(Y_test).describe())
    print(pd.DataFrame(X_test).describe())
    print("Mean squared error: ")
    print(np.mean(np.mean(np.power(regr24.predict(X_test) - Y_test,2),axis=0).reshape(24,7,order='C'),axis=0))
# Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % regr24.score(X_test, Y_test))

# try my original, not flattened, with scaling
    xscaler=StandardScaler().fit(target.df[0:8760])
    print(xscaler.mean_)
    print(xscaler.scale_)
    X_train=xscaler.transform(target.df[0:8760])
    X_test=xscaler.transform(target.df[8760:17496])
    Y_train=xscaler.transform(target.df.ix[48:8808,:])
    Y_test=xscaler.transform(target.df.ix[8808:17544, :])
    print(X_train.shape)
    print(X_test.shape)
    print(Y_train.shape)
    print(Y_test.shape)
    regrss = linear_model.LinearRegression()
    print(pd.DataFrame(X_train).describe())
    print(pd.DataFrame(Y_train).describe())
    regrss.fit(X_train, Y_train)
    print(pd.DataFrame(Y_test).describe())
    print(pd.DataFrame(X_test).describe())
    print('Coefficients: \n', regrss.coef_)
    print('Intercept \n', regrss.intercept_)   
    print("Mean squared error scaled: ")
    print(np.mean(np.power(regrss.predict(X_test) - Y_test,2),axis=0))
    print("Mean squared error: ")
    print(np.mean(np.power(xscaler.inverse_transform(regrss.predict(X_test)) - xscaler.inverse_transform(Y_test),2),axis=0))
# Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % regrss.score(X_test, Y_test))

    # now scale the every hour version as well! 
    bignew = np.array(target.df).reshape(731,168,order='C')
    print(bignew.shape)
    #    bignew[0:5,:]
    xscaler=StandardScaler().fit(bignew[0:365,:])
    print(xscaler.mean_)
    print(xscaler.scale_)
    X_train=xscaler.transform(bignew[0:365,:])
    X_test=xscaler.transform(bignew[365:729,:])
    Y_train=xscaler.transform(bignew[2:367,:])
    Y_test=xscaler.transform(bignew[367:731, :])
    print(X_train.shape)
    print(X_test.shape)
    print(Y_train.shape)
    print(Y_test.shape)
    regrss24 = linear_model.LinearRegression()
    print(pd.DataFrame(X_train).describe())
    print(pd.DataFrame(Y_train).describe())
    regrss24.fit(X_train, Y_train)
    print('Coefficients: \n', regrss24.coef_)
    print('Intercept \n', regrss24.intercept_)   
    print(pd.DataFrame(Y_test).describe())
    print(pd.DataFrame(X_test).describe())
    print("Mean squared error scaled: ")
    print(np.mean(np.mean(np.power(regrss24.predict(X_test) - Y_test,2),axis=0).reshape(24,7,order='C'),axis=0))
    print("Mean squared error: ")
    print(np.mean(np.mean(np.power(xscaler.inverse_transform(regrss24.predict(X_test)) - xscaler.inverse_transform(Y_test),2),axis=0).reshape(24,7,order='C'),axis=0))
# Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % regrss24.score(X_test, Y_test))


if __name__ == "__main__":
    main()
