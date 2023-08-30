import numpy
import streamlit
from matplotlib import pyplot

def MC(m, s, N):
  sims = numpy.random.normal(m, s, N)
  fig = pyplot.hist(sims, bins = int(numpy.sqrt(N)))
  streamlit.pyplot(fig)
  return numpy.mean(sims)

m = streamlit.number_input('mean') 
s = streamlit.number_input('stdev', min_value = 0)
N = streamlit.number_input('size', min_value = 1, step = 1)
print('empirical mean = ', MC(m, s, N))
