import rdrdois as rdr
import advancedreader as adv
import matplotlib.pyplot as plt
import numpy as np

g0simple=rdr.load_events('wave1_1min3s.txt')
g0=adv.processframe(g0simple)
g1simple=rdr.load_events('wave1_2min30s.txt')
g1=adv.processframe(g1simple)
g2simple=rdr.load_events('wave1_3min0s.txt')
g2=adv.processframe(g2simple)
g3simple=rdr.load_events('wave1_4min0s.txt')
g3=adv.processframe(g3simple)
g4simple=rdr.load_events('wave1_6min25s.txt')
g4=adv.processframe(g4simple)
g5simple=rdr.load_events('wave1_10min0s.txt')
g5=adv.processframe(g5simple)

Trasmus=[63*10**9,150*10**9,180*10**9,240*10**9,385*10**9,600*10**9]
Tdigitizer=[g0.Timestamp[len(g0)-1]-g0.Timestamp[0],
            g1.Timestamp[len(g1)-1]-g1.Timestamp[0],
            g2.Timestamp[len(g2)-1]-g2.Timestamp[0],
            g3.Timestamp[len(g3)-1]-g3.Timestamp[0],
            g4.Timestamp[len(g4)-1]-g4.Timestamp[0],
            g5.Timestamp[len(g5)-1]-g5.Timestamp[0]]
t = np.linspace(0, g5.Timestamp[len(g5)-1], 10000)
fit=np.polyfit(Tdigitizer,Trasmus,1)
plt.plot(t,fit[0]*t+fit[1],'--',Tdigitizer,Trasmus,'.')
plt.xlabel('digitizer timestamp $t_{digi}$')
plt.ylabel('manually measured time in ns')
textstr = 'fit=a$\cdot$$t_{digi}$+b\n$a=%.2f$\n$b=%.2f$' % (fit[0], fit[1])
plt.text(0.05, 500*10**9, textstr, fontsize=14,verticalalignment='top')
plt.show()

