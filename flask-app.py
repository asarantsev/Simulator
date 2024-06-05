from flask import Flask, render_template, request
import numpy
from matplotlib import pyplot
import os

app = Flask(__name__)
app.config["DEBUG"] = True
@app.route('/')

def index():
    if request.method == "GET":
        return render_template("main_page.html")

current_dir = os.path.abspath(os.path.dirname(__file__))

vixOut = [0.3603, 0.361, 0.3929, 0.4242, 0.4678, 0.4908, 0.5969, 0.6028, 0.6455, 0.76, 0.9572, 1.0844]
vixStat = {'slope' : -0.119, 'intercept' : 0.350, 'std' : 0.121, 'outliers': vixOut}
nlargeStat = {'intercept' : 3.3557, 'vix' : -0.1168, 'std' : 0.1941}
rlargeStat = {'intercept' : 3.1594, 'vix' : -0.1189, 'std' : 0.196}
nsmallStat = {'large' : 1.06, 'vix' : 0.000685, 'std' : 0.127, 'outliers' : [0.5849]}
rsmallStat = {'large' : 1.061, 'vix' : 0.001412, 'std' : 0.127, 'outliers' : [0.5857]}
NDATA = 459
NSIMS = 1000
NDISPLAYS = 5

def simVol(initial, horizon):
    prob = len(vixStat['outliers'])/NDATA
    N = numpy.random.normal(0, vixStat['std'], horizon)
    T = numpy.random.choice(vixStat['outliers'], horizon)
    Choice = numpy.random.uniform(0, 1, horizon)
    simLogRes = N * (Choice > prob) + T * (Choice < prob)
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
    res = N * (Choice > 1/NDATA) + stat['outliers'] * numpy.ones(horizon) * (Choice < 1/NDATA)
    return stat['large'] * simL + stat['vix'] * simVix[1:] + simVix[1:] * res

def simWealth(initialVolatility, initialWealth, adjustment, portfolio, flow, horizon):
    if adjustment == "Nominal":
        largeStat = nlargeStat
        smallStat = nsmallStat
    if adjustment == "Real":
        largeStat = rlargeStat
        smallStat = rsmallStat
    simVix = simVol(initialVolatility, horizon)
    simL = simLarge(largeStat, simVix, horizon)
    simS = simSmall(smallStat, simVix, simL, horizon)
    returns = simL * portfolio + simS * (1 - portfolio)
    timeAvgRet = 12*numpy.mean(returns)
    wealth = [initialWealth]
    for t in range(horizon):
        if (wealth[t] == 0):
            wealth.append(0)
        else:
            new = max(wealth[t] * (1 + returns[t]/100) + flow, 0)
            wealth.append(new)
    return timeAvgRet, numpy.array(wealth)

@app.route('/compute', methods=["GET", "POST"])

def compute():
    months = int(request.form['months'])
    years = int(request.form['years'])
    initialW = float(request.form['initialWealth'])
    portfolio = float(request.form['large'])
    cashFlowChoice = request.form.get('flowChoice')
    flow = float(request.form['flow'])
    indicatorsChoice = request.form.get('indicators')
    adjusted = request.form.get('adjusted')
    if (indicatorsChoice == 'user'):
        initialVol = float(request.form['initialVol'])
        textIndicators = 'User Given '
    elif (indicatorsChoice == 'average'):
        initialVol = 19.95
        textIndicators = 'Historical Average '
    elif (indicatorsChoice == 'current'):
        initialVol = 11
        textIndicators = 'Current Market '
    NSTEPS = months + 12 * years
    if (cashFlowChoice == "Nothing"):
        textFlow = 'No regular contributions or withdrawals'
        flows = 0
    elif (cashFlowChoice == "Contributions"):
        textFlow = 'Contributions ' + str(flow) + ' per month'
        flows = flow
    elif (cashFlowChoice == 'Withdrawals'):
        textFlow = 'Withdrawals ' + str(flow) + ' per month'
        flows = -flow
    paths = []
    timeAvgRets = []
    for sim in range(NSIMS):
        timeAvgRet, wealthSim = simWealth(initialVol, initialW, adjusted, portfolio, flows, NSTEPS)
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
    timeHorizonText = 'Time Horizon: ' + str(months) + ' months and ' + str(years) + ' years'
    initWealthText = 'Initial Wealth ' + str(round(initialW))
    Portfolio = 'The portfolio: ' + str(round(100*portfolio)) + '% large, ' + str(round(100*(1 - portfolio))) + '% small stocks'
    initMarketText = 'initial conditions: Volatility ' + str(round(initialVol, 2))
    SetupText = 'SETUP: ' + Portfolio + '\n' + timeHorizonText + '\n' + initWealthText +'\n' + textIndicators + initMarketText + '\n' + textFlow + '\n'
    if numpy.isnan(avgRet):
        ResultText = 'RESULTS: 100% Ruin Probability, always zero wealth'
    else:
        RuinProbText = str(round(100*ruinProb, 2)) + '% Ruin Probability'
        AvgRetText = 'time averaged annual returns:\naverage over all paths without ruin ' + str(round(avgRet, 2)) + '%'
        MeanText = 'average final wealth ' + str(round(wealthMean))
        MeanCompText = 'final wealth exceeds average with probability ' + str(round(100*meanProb, 2)) + '%'
        ResultText = 'RESULTS: ' + RuinProbText + '\n' + AvgRetText + '\n' + MeanText + '\n' + MeanCompText
    bigTitle = SetupText + '\n' + ResultText + '\n'
    pyplot.plot([0], [initialW], color = 'w', label = bigTitle)
    for display in range(NDISPLAYS):
        index = selectedIndices[display]
        rankText = ' final wealth, ranked ' + str(round(100*(2*display + 1)/(2*NDISPLAYS))) + '% '
        selectTerminalWealth = round(paths[index, -1])
        if (selectTerminalWealth == 0):
            pyplot.plot(times, paths[index], label = '0' + rankText + 'Gone Bust !!!')
        else:
            pyplot.plot(times, paths[index], label = str(selectTerminalWealth) + rankText + 'returns: ' + str(round(timeAvgRets[index], 2)) + '%')
    pyplot.xlabel('Years')
    pyplot.ylabel('Wealth')
    pyplot.title('Wealth Plot')
    pyplot.legend(bbox_to_anchor=(1.05, 1.05), loc='upper left', prop={'size': 12})
    #pyplot.rcParams['legend.title_fontsize'] = 'xx-large'
    image_path = os.path.join(current_dir, 'static', 'wealth.png')
    pyplot.savefig(image_path, bbox_inches='tight')
    pyplot.close()
    return render_template('main_page.html')
