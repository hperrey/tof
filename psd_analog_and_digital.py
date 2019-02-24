# coding: utf-8
Ddummy = D.query('0<ps<1 and channel==0 and amplitude>100 and qdc_lg<2250000')
Adummy = A.query('0<(qdc_det0-qdc_sg_det0)/qdc_det0 and qdc_det0<5000')
L=min(len(Adummy), len(Ddummy))
Ddummy=Ddummy.head(L)
Adummy=Adummy.head(L)
ax1=plt.subplot(1, 2, 1)
lg_offset = 65
sg_offset = 65
lg = Ddummy.qdc_lg + 500*lg_offset
sg = Ddummy.qdc_sg + 60*sg_offset
ps = (lg-sg)/lg

plt.hexbin(lg, ps, gridsize=80, cmap='inferno', label='digitized')
ax1=plt.subplot(1, 2, 2)
plt.hexbin(Adummy.qdc_det0, (Adummy.qdc_det0-Adummy.qdc_sg_det0)/Adummy.qdc_det0, gridsize=80, cmap='inferno', label='analog')
plt.show()
