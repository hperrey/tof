import pandas as pd
import matplotlib.pyplot as plt
N=pd.read_hdf('combined_no_waveforms.h5')

plt.hist(N.area, range=(0,12000), bins=1200, log=True, color='g',alpha=0.72); plt.xlabel('mv$\cdot$ns');plt.ylabel('Counts');plt.title('Charge spectrum, 5 mV theshold, %dpulses'%len(N));plt.show()
