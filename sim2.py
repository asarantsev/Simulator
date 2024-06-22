import numpy
from matplotlib import pyplot as plt

vixOut = [0.3603, 0.361, 0.3929, 0.4242, 0.4678, 0.4908, 0.5969, 0.6028, 0.6455, 0.76, 0.9572, 1.0844]
vixStat = {'slope' : -0.119, 'intercept' : 0.350, 'std' : 0.121, 'prob' : 0.026, 'outliers': vixOut}
nominalCov = [[0.03777, 0.03990], [0.03990, 0.05830]]
realCov = [[0.03852, 0.04072], [0.04072, 0.05921]]
nominal = {'const' : [3.3557, 3.6985], 'vix' : [-0.1168, -0.1311], 'cov' : nominalCov}
real = {'const' : [3.1594, 3.5021], 'vix' : [-0.1189, -0.1332], 'cov' : realCov}

M = 459
NSIMS = 1000
NSTEPS = 120
NDISPLAYS = 5
initialVol = 20
initialWealth = 1000
portfolio = 0.7
statistics = nominal
withdrawals = 4
timeAvgRets = []
paths = []

def simVol(initial, horizon):
    N = numpy.random.normal(0, vixStat['std'], horizon)
    T = numpy.random.choice(vixStat['outliers'], horizon)
    Choice = numpy.random.uniform(0, 1, horizon)
    simLogRes = N * (Choice > vixStat['prob']) + T * (Choice < vixStat['prob'])
    result = numpy.array([numpy.log(initial)])
    for t in range(horizon):
        current = result[-1]
        new = current * (vixStat['slope'] + 1) + vixStat['intercept'] + simLogRes[t]
        result = numpy.append(result, [new])
    return numpy.exp(result)

def simStocks(stat, simVix, horizon):
    res = numpy.random.multivariate_normal([0, 0], stat['cov'], horizon)
    lres = res[:, 0]
    sres = res[:, 1]
    sVix = simVix[1:]
    large = numpy.ones(horizon) * stat['const'][0] + sVix * stat['vix'][0] + sVix * lres
    small = numpy.ones(horizon) * stat['const'][1] + sVix * stat['vix'][1] + sVix * sres
    return (large, small)

def simWealth(initialV, initialW, flow):
    simVix = simVol(initialV, NSTEPS)
    simL, simS = simStocks(statistics, simVix, NSTEPS)
    returns = simL * portfolio + simS * (1 - portfolio)
    timeAvgRet = 12*numpy.mean(returns)
    wealth = [initialW]
    for t in range(NSTEPS):
        if (wealth[t] == 0):
            wealth.append(0)
        else:
            new = max(wealth[t] * (1 + returns[t]/100) + flow, 0)
            wealth.append(new)
    return timeAvgRet, numpy.array(wealth)

def graph(initialV, initialW, flow):
    paths = []
    for sim in range(NSIMS):
        timeAvgRet, wealthSim = simWealth(initialV, initialW, flow)
        timeAvgRets.append(timeAvgRet)
        paths.append(wealthSim)
    paths = numpy.array(paths)
    avgRet = numpy.mean([timeAvgRets[sim] for sim in range(NSIMS) if paths[sim, -1] > 0])
    wealthMean = numpy.mean(paths[:, -1])
    meanProb = numpy.mean([paths[sim, -1] > wealthMean for sim in range(NSIMS)])
    ruinProb = numpy.mean([paths[sim, -1] == 0 for sim in range(NSIMS)])
    sortedIndices = numpy.argsort(paths[:, -1])
    selectedIndices = [sortedIndices[int(NSIMS*(2*k+1)/(2*NDISPLAYS))] for k in range(NDISPLAYS)]
    times = [k/12 for k in range(NSTEPS + 1)]
    Portfolio = 'The portfolio is constant-weighted\n' 
    textIndicators = 'initial conditions:\n Volatility ' + str(round(initialVol, 2)) 
    Results = 'RESULTS ' + str(round(100*ruinProb, 2)) + '% Ruin Probability\n time averaged annual returns:\n average over all paths without ruin ' + str(round(avgRet, 2)) + '%'
    MeanResults = 'and average final wealth ' + str(round(wealthMean)) + '\nfinal wealth exceeds average with probability ' + str(round(100*meanProb, 2)) + '%'
    if numpy.isnan(avgRet):
        Results = '100% Ruin Probability'
        MeanResults = 'zero wealth always'
    TimeHorizon = 'Time Horizon is '
    bigTitle = 'SETUP: Initial Wealth ' + str(round(initialWealth)) + '\n' + TimeHorizon + '\n' + Portfolio + '\n' + textIndicators + '\n' + Results + '\n' + MeanResults
    for display in range(NDISPLAYS):
        index = selectedIndices[display]
        rankText = ' final wealth, ranked ' + str(round(100*(2*display + 1)/(2*NDISPLAYS))) + '% '
        selectTerminalWealth = round(paths[index, -1])
        if (selectTerminalWealth == 0):
            plt.plot(times, paths[index], label = '0' + rankText + 'Gone Bust !!!')
        else:
            plt.plot(times, paths[index], label = str(selectTerminalWealth))
    plt.xlabel('Years')
    plt.ylabel('Wealth')
    plt.title('Wealth Plot')
    plt.legend(title = bigTitle, bbox_to_anchor=(0.05, 1.05), loc='upper left')
    plt.show()

graph(initialVol, initialWealth, -withdrawals)