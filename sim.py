import numpy
import streamlit
from matplotlib import pyplot

def MC(m, s, N):
  sims = numpy.random.normal(m, s, N)
  pyplot.hist(sims, bins = int(numpy.sqrt(N)))
  pyplot.show()
  return numpy.mean(sims)

def main():
  m = input('Mean')
  s = input('Stdev')
  N = input('Sample')
  print('empirical mean = ', MC(m, s, N))
  return 0

if __name__ == "__main__":
    main()

