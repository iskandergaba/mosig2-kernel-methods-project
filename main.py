import preprocessor as pp

Xtrs = ['data/Xtr0.csv', 'data/Xtr1.csv', 'data/Xtr2.csv']
Ytrs = ['data/Ytr0.csv', 'data/Ytr1.csv', 'data/Ytr2.csv']
Xtes = ['data/Xte0.csv', 'data/Xte1.csv', 'data/Xte2.csv']
hot_code = {'A': '0', 'C': '1', 'G': '2', 'T': '3'}

pp.preprocess(Xtrs, Ytrs, Xtes, hot_code)