import numpy
import pandas
from scipy import stats
from statsmodels.api import OLS
from matplotlib import pyplot as plt
from statsmodels.graphics.gofplots import qqplot
from statsmodels.graphics.tsaplots import plot_acf

def test(data, label):
    qqplot(data, line = 's')
    plt.title(label)
    plt.show()
    plot_acf(data)
    plt.title('original ' + label)
    plt.show()
    plot_acf(abs(data))
    plt.title('absolute ' + label)
    plt.show()

# read data
DF = pandas.read_excel('data.xlsx', sheet_name = 'data')
M = len(DF) - 1
vix = DF['VIX'].values[1:]
small = DF['Small'].values[1:]
large = DF['Large'].values[1:]
cpi = DF['CPI'].values
inflation = 100*(cpi[1:]/cpi[:-1] - numpy.ones(M))
nlarge = large/vix
nsmall = small/vix
rlarge = (large - inflation)/vix
rsmall = (small - inflation)/vix

# model log VIX as AR(1)
lvix = numpy.log(vix)
regVix = stats.linregress(lvix[:-1], numpy.diff(lvix))
slopeVix = regVix.slope
interceptVix = regVix.intercept
vixRes = numpy.array([numpy.diff(lvix)[k] - slopeVix * lvix[k] - interceptVix for k in range(M-1)])
test(vixRes, 'volatility')
vixRes = sorted(vixRes)
outliers = vixRes[-12:]
meanVix = numpy.mean(vixRes[:-12])
stdVix = numpy.std(vixRes[:-12])
probVix = 12/M

# model large stock returns
regDF = pandas.DataFrame({'const' : 1/vix, 'vix' : 1})
nlargeReg = OLS(nlarge, regDF).fit()
print(nlargeReg.summary())
resnlarge = nlargeReg.resid
print('stderr nominal large = ', numpy.std(resnlarge))
test(resnlarge, 'nominal large')
rlargeReg = OLS(rlarge, regDF).fit()
print(rlargeReg.summary())
resrlarge = rlargeReg.resid 
print('stderr real large = ', numpy.std(resrlarge))
test(resrlarge, 'real large')

# model small stock returns
nsmallReg = OLS(nsmall, regDF).fit()
print(nsmallReg.summary())
resnsmall = nsmallReg.resid
test(resnsmall, 'nominal small')
rsmallReg = OLS(rsmall, regDF).fit()
print(rsmallReg.summary())
resrsmall = rsmallReg.resid
test(resrsmall, 'real small')

print('nominal residuals covariance matrix = ', numpy.cov([resnlarge, resnsmall]))
print('real residuals covariance matrix = ', numpy.cov([resrlarge, resrsmall]))