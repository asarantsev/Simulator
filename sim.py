import numpy
from matplotlib import pyplot as plt

vixOut = [0.3603, 0.361, 0.3929, 0.4242, 0.4678, 0.4908, 0.5969, 0.6028, 0.6455, 0.76, 0.9572, 1.0844]
vixStat = {'slope' : -0.119, 'intercept' : 0.350, 'std' : 0.121, 'prob' : 0.026, 'outliers': vixOut}
nlargeStat = {'intercept' : 3.3557, 'vix' : -0.1168, 'std' : 0.1941}
rlargeStat = {'intercept' : 3.1594, 'vix' : -0.1189, 'std' : 0.196}
nsmallStat = {'large' : 1.06, 'vix' : 0.000685, 'std' : 0.127, 'outliers' : [0.5849]}
rsmallStat = {'large' : 1.061, 'vix' : 0.001412, 'std' : 0.127, 'outliers' : [0.5857]}

M = 459
NSIMS = 1000
NSTEPS = 120
NDISPLAYS = 5
initialVol = 20
initialWealth = 1000
portfolio = 0.7
largeStat = rlargeStat
smallStat = rsmallStat
withdrawals = 8
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

def simLarge(stat, simVix, horizon):
    res = numpy.random.normal(0, stat['std'], horizon)
    return numpy.ones(horizon) * stat['intercept'] + simVix[1:] * stat['vix'] + simVix[1:] * res

def simSmall(stat, simVix, simL, horizon):
    N = numpy.random.normal(0, stat['std'], horizon)
    Choice = numpy.random.uniform(0, 1, horizon)
    res = N * (Choice > 1/M) + stat['outliers'] * numpy.ones(horizon) * (Choice < 1/M)
    return stat['large'] * simL + stat['vix'] * simVix[1:] + simVix[1:] * res

def simWealth(initialV, initialW, flow):
    simVix = simVol(initialV, NSTEPS)
    simL = simLarge(largeStat, simVix, NSTEPS)
    simS = simSmall(smallStat, simVix, simL, NSTEPS)
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