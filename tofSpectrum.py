Tlist=[]
i=0
for tn in ne213.TimeStamp:
    for tg in yap.TimeStamp:
        if -1500<tg-tn<1500:
            Tlist.append(tg-tn)
    if i%1000==0:
        print(i,'/',len(ne213))
    i+=1

plt.hist(Tlist,900,range=(-450,450));plt.show()
