import numpy as np
import pandas as pd
from mysqlconfig import cursor
import re
import plotly.offline as py
import plotly.graph_objects as go
import networkx as nx
import matplotlib.pyplot as plt
from main import get_unique
from main import ignite
import collections
import seaborn as sns #για ομορφιά
from networkx.algorithms import community
import time
#import pygraphviz as pgv
from plug import newplot
from random import seed
import random
import netcomp as nc
import math
import scipy.stats as stats
import netrd
#from pyensae.languages import r2python
from matplotlib.backends.backend_pdf import PdfPages
import deepgraph as dg
import sys
from portrait_divergence import portrait_divergence,portrait_divergence_weighted
from pyvis.network import Network
from IPython.core.display import display, HTML
from drawgraph import draw_graph3


def freedman_diaconis(data, returnas="width"):
    """
    Use Freedman Diaconis rule to compute optimal histogram bin width.
    ``returnas`` can be one of "width" or "bins", indicating whether
    the bin width or number of bins should be returned respectively.


    Parameters
    ----------
    data: np.ndarray
        One-dimensional array.

    returnas: {"width", "bins"}
        If "width", return the estimated width for each histogram bin.
        If "bins", return the number of bins suggested by rule.
    """
    data = np.asarray(data, dtype=np.float_)
    IQR  = stats.iqr(data, rng=(25, 75), scale=1.0, nan_policy="omit")
    N    = data.size
    bw   = (2 * IQR) / np.power(N, 1/3)

    if returnas=="width":
        result = bw
    else:
        datmin, datmax = data.min(), data.max()
        datrng = datmax - datmin
        result = int((datrng / bw) + 1)
    return(result)

def rtosnake():
    f = open("danai.txt", "w+")
    r = open("DeltaCon.R","r")
    rscript = r.read()
    print(r2python(rscript, pep8=True))
    g1 = open("graph1.txt","r")
    g2 = open("graph2.txt","r")

    f.write()
    f.close()

def controlroom(fromdate,todate,*args,category="None",restriction="None"):
    if args==():
        G=newplotting(fromdate,todate,category,restriction)
    elif args[0]=="country":
        countryaggr(fromdate,todate,category)

    #G=newplotting(fromdate,todate,category)
    #newplot(G)
    #print(nx.adjacency_matrix(G))
    #print(type(nx.adjacency_matrix(G)))
    #nx.write_edgelist(loseweight(G),"edgelist.txt")
    return G

#η γενική συνάρτηση στην οποία συμβαίνουν τα πάντα
def newplotting(fromdate,todate,category="None",restriction="None"):
    G = nx.DiGraph()
    trans = get_trans(fromdate, todate, restriction)
    #print("trans",trans)
    nodes=get_unique_nodes(fromdate, todate,category)
    #nodes=cleannodes(nodes,trans)#αφαιρεί κόμβους που δεν έχουν ακμή(χρειάζομαι connected graph)
    nodesup=[(i[0].upper(),i[1],i[2],i[3],i[4]) for i in nodes]#τα κάνω κεφαλαία
    transup=[(i[0].upper(),i[1].upper(),i[2],i[3]) for i in trans]#τα κάνω κεφαλαία
    transup = cleantrans(transup,nodesup) #backup κατάλοιπο για κόμβους που δεν υπάρχουν στο nodes
    #trans=thecleanest(trans,nodes)
    transup = cleantransloop(transup)
    nodesup = cleannodes(nodesup,transup)
    print("these are trans homie:",len(transup),"these dem nodes",len(nodesup))
    #testingfun(nodes,trans)
    #print(nodesup)
    #trans = cleantrans(transup,nodesup) #backup κατάλοιπο για κόμβους που δεν υπάρχουν στο trans
    G=nodify(nodesup,G)#φέρνω τους κόμβους σε καταλλληλη μορφή για το γράφημα
    G=edgify(transup,G)#φέρνω τις ακμές σε κατάλληλη μορφή για το γράφημα
    #newplot(G)#κάλεσμα συνάρτησης από το plug.py για το σχεδιασμό
    #nx.draw(G, with_labels=True, node_size=1500, alpha=0.3, arrows=True)
    #plt.show()
    #dg.plot_2d()
    return G
#συνάρτηση σαν την παραπάνω αλλά για την υλοποίηση node aggregation με βάση τη χώρα
def countryaggr(fromdate,todate):
    G = nx.DiGraph()
    trans = get_trans_country(fromdate, todate)
    nodes=get_unique_nodes(fromdate, todate)
    nodesup=[(i[0].upper(),i[1],i[2],i[3],i[4]) for i in nodes]
    transup=[(i[0].upper(),i[1].upper(),i[2],i[3]) for i in trans]
    #trans = cleantrans(transup,nodesup) #backup κατάλοιπο για κόμβους που δεν υπάρχουν στο trans
    G=nodifycountry(nodesup,G)
    G=edgify(transup,G)
    newplot(G)

def testingfun(nodes,trans):
    names = [item[0] for item in nodes]
    #print(names)
    #print(names.count('EUROPEAN COMMISSION'))

def cleannodes(nodes,trans):
    cleannodes=[]
    hashing=len(nodes)*[0]
    for j in range(len(trans)):
        for i in range(len(nodes)):
            if nodes[i][0] == trans[j][0] or nodes[i][0] == trans[j][1]:
                hashing[i]=1
    for i in range(len(nodes)):
        if hashing[i]==1:
            cleannodes.append(nodes[i])
    return cleannodes

def cleantransloop(trans):
    tran=[]
    for i in range(len(trans)):
        if trans[i][0]!=trans[i][1]:
            tran.append((trans[i][0],trans[i][1],trans[i][2],trans[i][3]))
    return tran


def cleantrans(trans,nodes):
    tran=[]
    counter=0
    hashing=len(trans)*[1]
    nodenames=[nodes[i][0] for i in range(len(nodes))]
    for j in range(len(trans)):
            if trans[j][0] not in nodenames or trans[j][1] not in nodenames:
                hashing[j]=0
    for i in range(len(trans)):
           if hashing[i]==1:
                tran.append(trans[i])
    #print("counter",counter)
    #print(tran)
    return tran

def cleanedges(edges,nodes):
    edgesss=[]
    hashing=len(edges)*[1]
    for j in range(len(edges)):
        if edges[j][0] not in nodes or edges[j][1] not in nodes:
            hashing[j]=0
    for i in range(len(edges)):
        if hashing[i]==1:
            edgesss.append(edges[i])
    return edgesss

def cleantrans2(trans,nodes):
    tran = []
    #make some transformations in order to search the list
    nodename=[]
    for i in nodes:
        nodename.append(i[0])
    """for i in range(len(trans)):
        if trans[i][0] == "European Commission":
            trans[i][0] = "EUROPEAN COMMISSION"
        elif trans[i][1] == "European Commission":
            trans[i][1]= "EUROPEAN COMMISSION"""""
    for i in range(len(trans)):
        if trans[i][0] =="European Commission" and trans[i][1] != "European Commission":
            tran.append(("EUROPEAN COMMISSION",trans[i][1],trans[i][2],trans[i][3]))
        elif trans[i][0] !="European Commission" and trans[i][1] == "European Commission":
            tran.append((trans[i][0],"EUROPEAN COMMISSION",trans[i][2],trans[i][3]))
        elif trans[i][0] =="European Commission" and trans[i][1] == "European Commission":
            tran.append(("EUROPEAN COMMISSION","EUROPEAN COMMISSION",trans[i][2],trans[i][3]))
        elif (trans[i][0] in nodename) or (trans[i][1] in nodename):
            tran.append(trans[i])
    #print(tran)
    return tran



def nodifycountry(nodes,G):
    #coloring={"regulated":"blue","financial":"green","governmental":"yellow"}
    node=[]
    for i in nodes:
        node.append((i[3], {"pos": (np.random.uniform(0, 10), np.random.uniform(0, 10))}))
    G.add_nodes_from(node)
    #print(node)
    return G

def nodify(nodes,G):
    #coloring={"regulated":"blue","financial":"green","governmental":"yellow"}
    node=[]
    for i in nodes:
        node.append((i[0], {"pos": locbycat(i[1])}))
    G.add_nodes_from(node)
    #print(node)
    return G

def edgify(trans,G):
    tran=[]
    for i in trans:
        tran.append((i[0],i[1],i[2]))
    G.add_weighted_edges_from(tran)
    #print("tran:",tran)
    return G

def locbycat(type):
    if type=="regulated":
        pos=(np.random.uniform(4.7, 5.3),np.random.uniform(0, 10))
    elif type=="financial":
        pos=(np.random.uniform(8.7, 9.1),np.random.uniform(0, 10))
    elif type=="governmental":
        pos=(np.random.uniform(0.9, 1.3),np.random.uniform(0, 10))
    else:
        pos=(np.random.uniform(0, 10),np.random.uniform(0, 10))
    return pos

def cleandate(date):#ημερομηνία σε λίστα [μέρα,μήνας,έτος]
    clean=date.split("/")
    return clean

def dirtydate(date):
    return f"""{date[0]}/{date[1]}/{date[2]}"""

def nullstuff(fromdate,todate):#give % of null-including transactions
    sql = f"""select transferringaccountholder, acquiringaccountholder, nbofunits, transactiontype
            from transactions_new where transactiondate 
            between '{fromdate}' and '{todate}'"""
    cursor.execute(sql)
    all = cursor.fetchall()
    sql = f"""select transferringaccountholder, acquiringaccountholder, nbofunits, transactiontype
            from transactions_new
            where ((transferringaccountholder is NULL) or (acquiringaccountholder is NULL))
            and transactiondate between '{fromdate}' and '{todate}'"""
    cursor.execute(sql)
    nulls = cursor.fetchall()
    return(len(nulls)*100/len(all))

def nullplot():#plot bar graph for null%
    refer = ('2014-01-1', '2014-02-1', '2014-03-1', '2014-04-1', '2014-05-1', '2014-06-1', '2014-07-1', '2014-08-1', '2014-09-1',
             '2014-10-1', '2014-11-1', '2014-12-1','2015-1-1')
    refer3= ('2015-01-1', '2015-02-1', '2015-03-1', '2015-04-1', '2015-05-1', '2015-06-1', '2015-07-1', '2015-08-1', '2015-09-1',
             '2015-10-1', '2015-11-1', '2015-12-1','2016-1-1')
    refer2 = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
    tab=[]
    for i in range(len(refer3)-1):
        tab.append(nullstuff(refer3[i],refer3[i+1]))
    df = pd.DataFrame({'null%': tab}, index=refer2)
    df.plot(kind='bar', subplots=True, figsize=(16, 8))
    plt.show()

def get_unique_nodes(fromdate,todate,category="None"):#return list of unique account holders [('node1,'category',
    if category == "None":
        addition = ""
    elif "," not in category:
        addition = f"and class.category = '{category}'"
    elif len(category.split(","))==2:
        things=category.split(",")
        addition = f"and (class.category = '{things[0]}' or class.category = '{things[1]}'"
    else:
        things=category.split(",")
        addition = f"and (class.category = '{things[0]}' or class.category = '{things[1]} or or class.category = '{things[2]}'"

    fromd = cleandate(fromdate)
    tod = cleandate(todate)
    sql = f"""select distinct tran.acquiringaccountholder, class.category, class.sector, acc.country, class.registry, trade.tradeSize, pol.emitSize,karpf.tradeSize
    from transactions_new as tran, eutl_accountholders as acc,eutl_accholderclassification as class,EUTL_AccHolderTransactingSize as trade,EUTL_AccHolderPollutingSize as pol,EUTL_AccHolderKarpfHolderSize as karpf
    where tran.acquiringaccountholder = acc.holdername
    and trade.holder = acc.rawcode
    and karpf.holder = acc.rawcode
    and pol.holder = acc.rawcode
    and acc.rawcode = class.holder
    {addition}
    and tran.transactiondate 
    between '{fromd[2]}-{fromd[1]}-{fromd[0]}' and '{tod[2]}-{tod[1]}-{tod[0]}'"""
    sql2 = f"""select distinct tran.transferringaccountholder, class.category, class.sector, acc.country, class.registry,trade.tradeSize, pol.emitSize ,karpf.tradeSize
    from transactions_new as tran, eutl_accountholders as acc,eutl_accholderclassification as class,EUTL_AccHolderTransactingSize as trade,EUTL_AccHolderPollutingSize as pol, EUTL_AccHolderKarpfHolderSize as karpf
    where tran.transferringaccountholder = acc.holdername
    and trade.holder = acc.rawcode
    and karpf.holder = acc.rawcode
    and pol.holder = acc.rawcode
    and acc.rawcode = class.holder
    {addition}
    and tran.transactiondate 
    between '{fromd[2]}-{fromd[1]}-{fromd[0]}' and '{tod[2]}-{tod[1]}-{tod[0]}'"""
    cursor.execute(sql)
    acq = cursor.fetchall()
    cursor.execute(sql2)
    tran = cursor.fetchall()
    all=get_unique(acq+tran)
    #print("nodes",all)
    return all

def getnodesalt(trans): #get nodes from transactions as list ["node1","node2"...]
    nodes=[]
    for i in trans:
        nodes.append(i[0])
        nodes.append(i[1])
    cleannodes= get_unique(nodes)
    return cleannodes

def get_trans(fromdate,todate,restriction="None"): #συναλλαγές μεταξύ δυο ημερομηνιών
    if restriction!="None":
        addition= f"and nbofunits>{restriction}"
    else: addition=""
    fromd=cleandate(fromdate)
    tod=cleandate(todate)
    key = 2 #1 for everything in, 2 for excluding NULL,
    if key ==1 :
        sql = f"""select transferringaccountholder, acquiringaccountholder, nbofunits, transactiontype
        from transactions_new where
        transactiondate 
        between '{fromd[2]}-{fromd[1]}-{fromd[0]}' and '{tod[2]}-{tod[1]}-{tod[0]}'
        {addition}"""
    #If we want to exclude NULL appearances do the following
    elif key==2 :
        sql = f"""select transferringaccountholder, acquiringaccountholder, nbofunits, transactiontype
        from transactions_new
        where ((transferringaccountholder is not NULL) and (acquiringaccountholder is not NULL))
        and transactiondate between '{fromd[2]}-{fromd[1]}-{fromd[0]}' and '{tod[2]}-{tod[1]}-{tod[0]}'
        {addition}"""
    cursor.execute(sql)
    tran=cursor.fetchall()
    return tran

def get_trans_country(fromdate,todate):
    fromd = cleandate(fromdate)
    tod = cleandate(todate)
    key = 2  # 1 for everything in, 2 for excluding NULL,
    if key == 1:
        sql = f"""select TransferringRegistry,AcquiringRegistry, nbofunits, transactiontype
            from transactions_new where transactiondate 
            between '{fromd[2]}-{fromd[1]}-{fromd[0]}' and '{tod[2]}-{tod[1]}-{tod[0]}'"""
    # If we want to exclude NULL appearances do the following
    elif key == 2:
        sql = f"""select TransferringRegistry,AcquiringRegistry, nbofunits, transactiontype
            from transactions_new
            where ((transferringaccountholder is not NULL) and (acquiringaccountholder is not NULL))
            and transactiondate between '{fromd[2]}-{fromd[1]}-{fromd[0]}' and '{tod[2]}-{tod[1]}-{tod[0]}'"""
    cursor.execute(sql)
    tran = cursor.fetchall()
    return tran

def graphify(dic):
    df = pd.DataFrame(dic)
    G = nx.from_pandas_edgelist(df, 'from', 'to', create_using=nx.DiGraph())
    nx.draw(G, with_labels=False, node_size=1500, alpha=0.3, arrows=True)

def gov(fromdate,todate):#μοναδικοί governmental κόμβοι
    fromd = cleandate(fromdate)
    tod = cleandate(todate)
    sql = f"""select tran.acquiringaccountholder, class.category, class.sector, acc.country, class.registry
    from transactions_new as tran, eutl_accountholders as acc,eutl_accholderclassification as class
    where tran.acquiringaccountholder = acc.holdername
    and acc.rawcode = class.holder
    and class.category='governmental'
    and tran.transactiondate 
    between '{fromd[2]}-{fromd[1]}-{fromd[0]}' and '{tod[2]}-{tod[1]}-{tod[0]}'"""
    sql2 = f"""select tran.transferringaccountholder, class.category, class.sector, acc.country, class.registry
    from transactions_new as tran, eutl_accountholders as acc,eutl_accholderclassification as class
    where tran.transferringaccountholder = acc.holdername
    and acc.rawcode = class.holder
    and class.category='governmental'
    and tran.transactiondate 
    between '{fromd[2]}-{fromd[1]}-{fromd[0]}' and '{tod[2]}-{tod[1]}-{tod[0]}'"""
    cursor.execute(sql)
    acq = cursor.fetchall()
    cursor.execute(sql2)
    tran = cursor.fetchall()
    allgov=get_unique(acq+tran)
    return allgov

def fin(fromdate,todate):#μοναδικοί financial κόμβοι
    fromd = cleandate(fromdate)
    tod = cleandate(todate)
    sql = f"""select tran.acquiringaccountholder, class.category, class.sector, acc.country, class.registry
    from transactions_new as tran, eutl_accountholders as acc,eutl_accholderclassification as class
    where tran.acquiringaccountholder = acc.holdername
    and acc.rawcode = class.holder
    and class.category='financial'
    and tran.transactiondate 
    between '{fromd[2]}-{fromd[1]}-{fromd[0]}' and '{tod[2]}-{tod[1]}-{tod[0]}'"""
    sql2 = f"""select tran.transferringaccountholder, class.category, class.sector, acc.country, class.registry
    from transactions_new as tran, eutl_accountholders as acc,eutl_accholderclassification as class
    where tran.transferringaccountholder = acc.holdername
    and acc.rawcode = class.holder
    and class.category='regulated'
    and tran.transactiondate 
    between '{fromd[2]}-{fromd[1]}-{fromd[0]}' and '{tod[2]}-{tod[1]}-{tod[0]}'"""
    cursor.execute(sql)
    acq = cursor.fetchall()
    cursor.execute(sql2)
    tran = cursor.fetchall()
    allfin=get_unique(acq+tran)
    return allfin

def reg(fromdate,todate):#μοναδικοί regulated κόμβοι
    fromd = cleandate(fromdate)
    tod = cleandate(todate)
    sql = f"""select tran.acquiringaccountholder, class.category, class.sector, acc.country, class.registry
    from transactions_new as tran, eutl_accountholders as acc,eutl_accholderclassification as class
    where tran.acquiringaccountholder = acc.holdername
    and acc.rawcode = class.holder
    and class.category='financial'
    and tran.transactiondate 
    between '{fromd[2]}-{fromd[1]}-{fromd[0]}' and '{tod[2]}-{tod[1]}-{tod[0]}'"""
    sql2 = f"""select tran.transferringaccountholder, class.category, class.sector, acc.country, class.registry
    from transactions_new as tran, eutl_accountholders as acc,eutl_accholderclassification as class
    where tran.transferringaccountholder = acc.holdername
    and acc.rawcode = class.holder
    and class.category='regulated'
    and tran.transactiondate 
    between '{fromd[2]}-{fromd[1]}-{fromd[0]}' and '{tod[2]}-{tod[1]}-{tod[0]}'"""
    cursor.execute(sql)
    acq = cursor.fetchall()
    cursor.execute(sql2)
    tran = cursor.fetchall()
    allfin=get_unique(acq+tran)
    return allfin



def plotting(fromdate,todate):
    dic={'from':[],'to':[]}
    trans=get_trans(fromdate,todate)
    #transdf=pd.DataFrame(trans,columns=['acq','tran','units','type'])
    point01=time.time()
    print("point01 ", point01-start)
    #οι 2 παρακάτω γραμμές χρειάζονται για το degree distribution
    #dicoutnodes=dicify(trans)
    #dicinnodes=revdicify(trans)
    point02=time.time()
    print("point02",point02-point01)
    get_unique_nodes(fromdate,todate)
    nodes=getnodesalt(trans)
    print(len(gov(fromdate,todate)))
    #onlygov=cleantrans(trans,get_unique((fin(fromdate,todate)+reg(fromdate,todate))))
    onlyfin=cleantrans(trans,get_unique((gov(fromdate,todate)+reg(fromdate,todate))))
    #onlyreg=cleantrans(trans,get_unique((gov(fromdate,todate)+fin(fromdate,todate))))
    #nogovtrans=cleantrans(trans,gov(fromdate,todate))
    point03= time.time()
    print("point03",point03-point02)
    #nodes=get_unique_nodes(fromdate,todate)
    for i in range(0,len(trans)):
        dic['from'].append(trans[i][0])
        dic['to'].append(trans[i][1])
    df=pd.DataFrame(dic)
    point04=time.time()
    print("point04 ",point04-point03)
    #G = nx.from_pandas_edgelist(df, 'from', 'to', create_using=nx.DiGraph())
    #nx.draw(G, with_labels=True, node_size=1500, alpha=0.3, arrows=True)
    #plt.show()
    #degreedistr(dicoutnodes) #για κατανομή βαθμών κορυφών έσω
    #degreedistr(dicinnodes) #για κατανομή βαθμών κορυφών έξω
    mat=getmatrix(onlyfin,nodes)
    simplematrix = getsimplematrix(mat)
    point1= time.time()
    print("point1 ",point1-point04)
    G = nx.DiGraph(mat)
    point2= time.time()
    print("point2 ",point2-point1)
    #density = nx.density(G)
    #nx.draw(G, with_labels=True, node_size=1500, alpha=0.3, arrows=True)
    nx.draw_kamada_kawai(G,alpha=0.2)
    #newplot(G)
    point3=time.time()
    print("point3 ",point3-point2)
    plt.show()
    centr=nx.betweenness_centrality(G, k=None, normalized=True, weight=None, endpoints=False, seed=None)
    pagerank=nx.pagerank(G,alpha=0.8)
    #communities = community.greedy_modularity_communities(G)

def getmatrix(trans,nodes):
    mat = np.random.randint(0, 1, size=(len(nodes), len(nodes)))
    cleannodes=nodes
    for i in range(0,len(nodes)):
        for j in range(0,len(trans)):
            if nodes[i] == trans[j][0]:
                mat[i][cleannodes.index(trans[j][1])] += (trans[j][2])
    return mat

def getsimplematrix(matrix): #για adjacency matrix weighted, επιστρέφει τον απλό
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            if matrix[i][j]!=0:
                matrix[i][j]=1
    return matrix

def dicify(trans):
    dicnodes={}
    for i in trans:
        dicnodes[i[0]]=[]
        for j in trans:
            if i[0]==j[0]:
                dicnodes[i[0]].append(j[1])
    return dicnodes

def revdicify(trans):
    dicnodes = {}
    for i in trans:
        dicnodes[i[1]] = []
        for j in trans:
            if i[1] == j[1]:
                dicnodes[i[1]].append(j[0])
    return dicnodes

def degreedistr(dic):
    degrees=[len(dic[k]) for k in dic.keys()]
    degrees.sort()
    df = pd.DataFrame(degrees, columns=['degrees'])
    sns.distplot(df) #another way of visualizing

    #df.plot(kind='hist', subplots=False, figsize=(16, 8))
    #plt.hist(degrees)
    plt.show()
    #print(degrees)



def getdensity():
    "something"

def nodeaggr(fromdate,todate,aggr):
    fromd = cleandate(fromdate)
    tod = cleandate(todate)
    dic={'from':[],'to':[]}
    if aggr=="sector":
        key=2
    elif aggr == "category":
        key=1
    elif aggr == "country":
        key =3
    elif aggr == "countrypollution":
        key=4
    sql0=f"""select tran.transferringaccountholder, class.category, class.sector,acc.country, class.registry
        from transactions_new as tran, eutl_accountholders as acc,eutl_accholderclassification as class
        where tran.transferringaccountholder = acc.holdername
        and acc.rawcode = class.holder
        and tran.transactiondate 
        between '{fromd[2]}-{fromd[1]}-{fromd[0]}' and '{tod[2]}-{tod[1]}-{tod[0]}'"""
    sql1 = f"""select tran.acquiringaccountholder, class.category, class.sector,acc.country, class.registry
       from transactions_new as tran, eutl_accountholders as acc,eutl_accholderclassification as class
       where tran.acquiringaccountholder = acc.holdername
       and acc.rawcode = class.holder
       and tran.transactiondate 
       between '{fromd[2]}-{fromd[1]}-{fromd[0]}' and '{tod[2]}-{tod[1]}-{tod[0]}'"""
    cursor.execute(sql1)
    acq = cursor.fetchall()
    cursor.execute(sql0)
    tran = cursor.fetchall()
    all = get_unique(acq + tran)
    for i in range(0, len(tran)):
        dic['from'].append(tran[i][key])
        dic['to'].append(acq[i][key])
    df = pd.DataFrame(dic)
    G = nx.from_pandas_edgelist(df, 'from', 'to', create_using=nx.DiGraph())
    nx.draw(G, with_labels=True, node_size=1500, alpha=0.3, arrows=True)
    plt.show()

def yearly():
    refer=["01/01/14","01/02/14","01/03/14","01/04/14","01/05/14","01/06/14","01/07/14","01/08/14","01/09/14","01/10/14","01/11/14","01/12/14","01/01/15"]
    refer2=["JAN","FEB","MAR","APR","MAY","JUN","JUL","AUG","SEP","OCT","NOV","DEC"]
    for i in range(len(refer)-1):
        print(refer2[i])
        plotting(refer[i],refer[i+1])

def degreedist(G):
    degree_sequence = sorted([d for n, d in G.degree()], reverse=True)  # degree sequence
    degreeCount = collections.Counter(degree_sequence)
    deg, cnt = zip(*degreeCount.items())

    fig, ax = plt.subplots()
    plt.bar(deg, cnt, width=0.80, color="b")

    plt.title("Degree Histogram")
    plt.ylabel("Count")
    plt.xlabel("Degree")
    ax.set_xticks([d + 0.4 for d in deg])
    ax.set_xticklabels(deg)
    plt.show()

def getstats(fromdate,todate):
    G = controlroom(fromdate,todate)
    H = G.to_undirected()
    #newplot(G)
    #print("clustering UD",nx.clustering(G))
    #print("clustering UU",nx.clustering(H))
    print("average clustering UD",nx.average_clustering(G),file=f)
    print("average clustering UU",nx.average_clustering(H),file=f)
    print("average clustering WD",nx.average_clustering(G,weight='weight'),file=f)
    print("average clustering WU",nx.average_clustering(H,weight='weight'),file=f)
    #print(nx.average_clustering(G))
    #print("between",nx.betweenness_centrality(H))
    #bet=nx.betweenness_centrality(H)
    bet1=nx.betweenness_centrality(loseweight(H))
    #bet2=nx.betweenness_centrality(H,'weight')
    #print("betweenness_centrality",bet)
    #print("between", bet==bet1)
    #print("between2", bet1==bet2)
    #plt.hist(bet1, bins=1000)
    #plt.show()
    #print(dict(sorted(bet1.items(), key=lambda item: item[1])))
    #degreedist(G)
    #print("connected",nx.is_strongly_connected(G))
    #smallest = min(nx.strongly_connected_components(G), key=len)
    #print(type(smallest))
    #print(smallest)
    #print(G.edges("TEC SVISHTOV AD"))
    #print(G.edges("LECHWERKE AG"))
    #rint(list(nx.isolates(G)))
    #newplot(smallest)
    #int(nx.is_connected(H))
    #int("diameter",nx.diameter(H))
    #print("what the fuck is happening",list(nx.isolates(G)))
    #print(G.edges("DEPARTMENT FOR BUSINESS, ENERGY & INDUSTRIAL STRATEGY"))
    #print(G.edges("PGE POLSKA GRUPA ENERGETYCZNA S.A."))
    #print(G.edges())
    A = nx.adjacency_matrix(G)
    AU = nx.adjacency_matrix(H)
    #communities_generator = community.girvan_newman(G)
    #top_level_communities = next(communities_generator)
    #next_level_communities = next(communities_generator)
    #comms=[len(i) for i in sorted(map(sorted, next_level_communities))]
    #print("comms",comms)
    #print("hello again",sorted(map(sorted, next_level_communities)))
    assort=nx.degree_pearson_correlation_coefficient(G,x='out',weight='weight')
    assort2=nx.degree_pearson_correlation_coefficient(G,x='out',weight='None')#weight maybe should be something
    assort3=nx.degree_pearson_correlation_coefficient(H,x='out',weight='weight')
    assort4 = nx.degree_pearson_correlation_coefficient(H, x='out', weight='None')
    print("assortativity out","WD", assort,"UD", assort2,'WU',assort3,'UU',assort4,file=f)
    assort1 = nx.degree_pearson_correlation_coefficient(G, x='in', weight='weight')
    assort12 = nx.degree_pearson_correlation_coefficient(G, x='in', weight='None')  # weight maybe should be something
    #assort13 = nx.degree_pearson_correlation_coefficient(H, x='in', weight='weight')
    #assort14 = nx.degree_pearson_correlation_coefficient(H, x='in', weight='None')
    print("assortativity in","WD", assort1, "UD", assort12,file=f)
    #print(G.edges())
    #print(G.get_edge_data('TOPLOFIKACIA SOFIA EAD', 'EUROPEAN COMMISSION'))

def compar(fromdate1,todate1,fromdate2,todate2):
    G1 = controlroom(fromdate1,todate1)
    G2 = controlroom(fromdate2,todate2)
    H1 = loseweight(G1)
    H2 = loseweight(G2)
    G3 = nx.to_undirected(H1)
    G4 = nx.to_undirected(H2)
    if G3.number_of_nodes()>G4.number_of_nodes():
        dif=G3.number_of_nodes()-G4.number_of_nodes()
        listofnodes = G3.nodes()
        listofedges = G3.edges()
        RandomSample = random.sample(listofnodes, dif)
        print("difference", dif)
        L1 = nx.Graph()
        nodes = listofnodes - RandomSample
        print("these dem nodes", type(nodes))
        print("these dem edges", type(listofedges))
        L1.add_nodes_from(nodes)
        L1.add_edges_from(cleanedges(list(listofedges), nodes))
        print("I made it this far")
        print("this better be nice", L1.number_of_nodes())
        print(G4.number_of_nodes(), G3.number_of_nodes())
        L2 = G4
        print(L1.number_of_nodes(), L2.number_of_nodes())
    elif G3.number_of_nodes()<G4.number_of_nodes():
        dif=G4.number_of_nodes()-G3.number_of_nodes()
        listofnodes = G4.nodes()
        listofedges = G4.edges()
        RandomSample = random.sample(listofnodes, dif)
        print("difference",dif)
        L1 = nx.Graph()
        nodes=listofnodes - RandomSample
        print("these dem nodes",type(nodes))
        print("these dem edges",type(listofedges))
        L1.add_nodes_from(nodes)
        L1.add_edges_from(cleanedges(list(listofedges),nodes))
        print("I made it this far")
        print("this better be nice",L1.number_of_nodes())
        print(G4.number_of_nodes(),G3.number_of_nodes())
        L2=G3
        print(L1.number_of_nodes(),L2.number_of_nodes())
    #H1= nx.to_undirected(H1)
    #H2 = nx.to_undirected(H2)
    A1, A2 = [nx.to_numpy_array(G) for G in [L1, L2]]
    A3, A4 = [nx.to_numpy_array(G) for G in [H1, H2]]
    #dis=dist(G3,G4)
    #print(dis)
    d=netrd.distance.DeltaCon().dist(L1,L2,exact=True)
    d1=nc.deltacon0(A1,A2)
    #d1=netrd.distance.DeltaCon().dist(H1,H2)
    nottt=nc.deltacon0(A3,A3)
    print("samies",nottt)
    print("deltacon", d,d1)

def runever(fromdate,todate):
    comparison=[]
    cleanfrom=fromdate.split("/")
    cleanto=todate.split("/")
    cleanfrom2=fromdate.split("/")
    cleanto2=todate.split("/")
    for i in range(10):
        cleanfrom[2] = str(int(cleanfrom[2])+1)
        cleanto[2] = str(int(cleanfrom[2]) + 1)
        cleanfrom2[2] = str(int(cleanfrom[2])+1)
        cleanto2[2] = str(int(cleanfrom[2]) + 1)
        for j in range(3):
            cleanfrom[1] = str(4*(j+1))
            cleanto[1] = str(int(4*(j+1) +1))
            cleanfrom2[1] = str(int(cleanfrom[1]) + 1)
            cleanto2[1] = str(int(cleanfrom[1]) + 1)
            comparison.append(compar(f"{dirtydate(cleanfrom)}",f"{dirtydate(cleanto)}",f"{dirtydate(cleanfrom2)}",f"{dirtydate(cleanto2)}"))
        print('{0:2d} {1:3d} {2:4d} {3:5d}'.format(i, i * i, i * i * i,i))
        print(f"{cleanfrom[1]}")
        comparison=[]

def weightdist(fromdate,todate,category="None",restriction="None"):
    print(fromdate,todate)
    G=controlroom(fromdate,todate,category=category,restriction=restriction)
    listing=[edge for edge in G.edges()]
    weights=[]
    for edge in listing:
        weights.append(math.log(G[edge[0]][edge[1]]["weight"]))
    #print(weights)
    NBR_BINS = freedman_diaconis(weights, returnas="bins")
    density = stats.gaussian_kde(weights)
    n, x, _ = plt.hist(weights, bins=NBR_BINS,
                       histtype='bar', density=True)
    #print(n,type(n))
    #print(x,type(x))
    #print(weights)
    lab=str(cleandate(fromdate)[1])+"/"+str(cleandate(fromdate)[2])
    plt.plot(x,density(x),label=lab)
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
               ncol=2, mode="expand", borderaxespad=0.)
    plt.savefig(pp, format='pdf')
    # #plt.savefig('line_plot.png')
    #plt.show()
    return density(x)
    #degreedist(G)
    #for edge in G.edges:

def degdist(fromdate,todate,category="None",restriction="None"):
    G=controlroom(fromdate,todate,category=category,restriction=restriction)
    listing=[degree[1] for degree in G.degree()]
    listing2=[]
    #for elem in listing:
    #    if elem>0:
    #        listing2.append(math.log(elem))
    #listing2=[math.log(elem) for elem in listing]

    #weights=[]
    #for edge in listing:
    #    weights.append(math.log(G[edge[0]][edge[1]]["weight"]))
    #print(weights)
    density = stats.gaussian_kde(listing)
    n, x, _ = plt.hist(listing, bins=2*(int(len(listing)**(1/3))),
                       histtype='bar', density=True)
    #print(n,type(n))
    #print(x,type(x))
    #print(listing)
    plt.plot(x,density(x))
    plt.show()
    return density(x)
    #degreedist(G)
    #for edge in G.edges:


def loseweight(G):
    H = nx.DiGraph()
    for edge in G.edges():
        H.add_edge(edge[0],edge[1])
    return H

def multiple(fromdate,todate):
    pp = PdfPages('weights.pdf')
    fro=cleandate(fromdate)
    tod=cleandate(todate)
    densities=[]
    F=["1","4","2010"]
    T=["28","4","2010"]
    for i in range(4):
        plt.clf()
        for j in range(3):
            weightdist(dirtydate(F),dirtydate(T))
            F[1]=str((int(F[1])+4)%13)
            T[1]=str((int(T[1])+4)%13)
        F[1]=T[1]="4"
        F[2]=str(int(F[2])+1)
        T[2]=str(int(T[2])+1)
def mult(fromdate,todate):
    pp = PdfPages('weights.pdf')
    fro = cleandate(fromdate)
    tod = cleandate(todate)
    densities = []
    F = ["1", "4", "2010"]
    T = ["28", "4", "2010"]
    for i in range(4):
        for j in range(3):
            weightdist(dirtydate(F), dirtydate(T))
            F[1] = str((int(F[1]) + 4) % 13)
            T[1] = str((int(T[1]) + 4) % 13)
        F[1] = T[1] = "4"
        F[2] = str(int(F[2]) + 1)
        T[2] = str(int(T[2]) + 1)

def massstats(fromdate,todate):
    months=["January","February","March","April","May","June","July","August","September","October","November","December"]
    F=T=["1","1","2005"]
    dates=[("1/1/","31/1/"),("1/2/","28/2/"),("1/3/","31/3/"),("1/4/","30/4/"),("1/5/","31/5/"),("1/6/","30/6/"),("1/7/","31/7/"),("1/8/","31/8/")
        ,("1/9/","30/9/"),("1/10/","31/10/"),("1/11/","30/11/"),("1/12/","31/12/")]
    for i in range(10):
        year=2006+i
        print(year,file=f)
        for j in range(12):
            print(months[j],file=f)
            getstats(dates[j][0]+str(year),dates[j][1]+str(year))

def edli(fromdate,todate):
    months=["January","February","March","April","May","June","July","August","September","October","November","December"]
    dates = [("1/1/", "31/1/"), ("1/2/", "28/2/"), ("1/3/", "31/3/"), ("1/4/", "30/4/"), ("1/5/", "31/5/"),
             ("1/6/", "30/6/"), ("1/7/", "31/7/"), ("1/8/", "31/8/")
        , ("1/9/", "30/9/"), ("1/10/", "31/10/"), ("1/11/", "30/11/"), ("1/12/", "31/12/")]
    for i in range(4):
        year=2010+i
        for j in range(3):
            mont=4*(j+1)
            date=months[mont-1]+str(year)
            G=controlroom(dates[mont-1][0]+str(year),dates[mont-1][1]+str(year))
            H=nx.to_undirected(G)
            #f=open(f"{date}.txt","w+")
            nx.write_edgelist(G, f"{date}_directed.txt")
            nx.write_edgelist(H, f"{date}_undirected.txt")

def bigbois():
    "adsf"


def newcomp(fromdate1,todate1,fromdate2,todate2):
    G1 = controlroom(fromdate1, todate1)
    G2 = controlroom(fromdate2, todate2)
    H1 = loseweight(G1)
    H2 = loseweight(G2)
    G3 = nx.to_undirected(H1)
    G4 = nx.to_undirected(H2)
    dif = G3.number_of_nodes() - G4.number_of_nodes()
    listofnodes1 = G3.nodes()
    listofedges1 = G3.edges()
    listofnodes2 = G4.nodes()
    listofedges2 = G4.edges()
    nodes=list(set(listofnodes1).intersection(listofnodes2))

    L1 = nx.Graph()
    #print("these dem nodes", type(nodes))
    #print("these dem edges", type(listofedges))
    L1.add_nodes_from(nodes)
    L1.add_edges_from(cleanedges(list(listofedges1), nodes))
    print("I made it this far")
    print("this better be nice", L1.number_of_nodes())
    L2 = nx.Graph()
    L2.add_nodes_from(nodes)
    L2.add_edges_from(cleanedges(list(listofedges2),nodes))
    print("L1 weights",nx.is_weighted(L1),"L2 weights",nx.is_weighted(L2))
    print("L1 dir",nx.is_directed(L1),"L2 dir",nx.is_directed(L2))

    print("nodes comp",L1.number_of_nodes(), L2.number_of_nodes())
    # H1= nx.to_undirected(H1)
    # H2 = nx.to_undirected(H2)
    A1, A2 = [nx.to_numpy_array(G) for G in [L1, L2]]
    A3, A4 = [nx.to_numpy_array(G) for G in [H1, H2]]
    # dis=dist(G3,G4)
    # print(dis)
    d = netrd.distance.DeltaCon().dist(L1, L2)
    d1 = nc.deltacon0(A1, A2)
    # d1=netrd.distance.DeltaCon().dist(H1,H2)
    nottt = nc.deltacon0(A3, A3)
    print("samies", nottt)
    print("deltacon", d, d1)

def testin():
    G1=nx.Graph()
    G2 = nx.Graph()
    edges1=[("UK","Denmark"),("UK","Germany"),("UK","Spain"),("UK","Greece")]
    edges2=[("UK","Denmark"),("UK","Spain"),("UK","France"),("UK","Germany"),("Germany","Greece")]
    G1.add_edges_from(edges1)
    G1.add_node("France")
    print(list(G1.nodes))
    G2.add_edges_from(edges2)
    G2.add_edge("UK","France")
    print("printing distance")
    #d = netrd.distance.DeltaCon().dist(G1, G1)
    d = portrait_divergence(G1,G2)
    d1 = portrait_divergence_weighted(G1,G2)
    print(d,d1)

def toleda(month,year):
    months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
              "November", "December"]
    dates = [("1/1/", "31/1/"), ("1/2/", "28/2/"), ("1/3/", "31/3/"), ("1/4/", "30/4/"), ("1/5/", "31/5/"),
             ("1/6/", "30/6/"), ("1/7/", "31/7/"), ("1/8/", "31/8/")
        , ("1/9/", "30/9/"), ("1/10/", "31/10/"), ("1/11/", "30/11/"), ("1/12/", "31/12/")]
    minas=months.index(month)
    print("minas",minas)
    print(dates[minas][0]+str(year),dates[minas][1]+str(year))
    G=controlroom(dates[minas][0]+str(year),dates[minas][1]+str(year))
    nodes=list(G.nodes)
    nodesindex=[nodes.index(node) for node in nodes]
    print(nodesindex)
    edges=list(G.edges)
    edgesindex=[]
    for edge in edges:
        edgesindex.append((nodes.index(edge[0]),nodes.index(edge[1])))
    print(edgesindex)
    print("wegotthisfar")
    #print("how bout dem nodes",nodes,"how bout dem edges",edges)
    f = open(f"{month}{year}.gw", "w")
    print("#header section","LEDA.GRAPH","string","void","-1","#nodes section",len(nodesindex),sep='\n',file=f)
    for i in nodesindex:
        print("|{"+str(i)+"}|",file=f)
    print("\n", file = f)
    print("#edges section",len(edgesindex),sep="\n",file=f)
    for j in range(len(edgesindex)):
        print(edgesindex[j][0],edgesindex[j][1],"0","|{}|",sep=" ",file = f)
    f.close()

def edgeli(month,year):
    months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
              "November", "December"]
    dates = [("1/1/", "31/1/"), ("1/2/", "28/2/"), ("1/3/", "31/3/"), ("1/4/", "30/4/"), ("1/5/", "31/5/"),
             ("1/6/", "30/6/"), ("1/7/", "31/7/"), ("1/8/", "31/8/")
        , ("1/9/", "30/9/"), ("1/10/", "31/10/"), ("1/11/", "30/11/"), ("1/12/", "31/12/")]
    minas = months.index(month)
    print("minas", minas)
    print(dates[minas][0] + str(year), dates[minas][1] + str(year))
    G = controlroom(dates[minas][0] + str(year), dates[minas][1] + str(year))
    edges=list(G.edges)
    f = open(f"edgelist-{month}{year}.gw", "w")
    for edge in edges:
        print(edge[0],edge[1],file=f)
    f.close()

def jacc(fromdate1,todate1,fromdate2,todate2):
    G1 = controlroom(fromdate1, todate1)
    G2 = controlroom(fromdate2, todate2)
    H1 = loseweight(G1)
    H2 = loseweight(G2)
    G3 = nx.to_undirected(H1)
    G4 = nx.to_undirected(H2)
    edges1=list(G3.edges)
    edges2=list(G4.edges)
    jacc=netrd.distance.JaccardDistance().dist(G3,G4)
    print(jacc)

def lsd(fromdate1,todate1,fromdate2,todate2):
    G1 = controlroom(fromdate1, todate1)
    G2 = controlroom(fromdate2, todate2)
    H1 = loseweight(G1)
    H2 = loseweight(G2)
    G3 = nx.to_undirected(H1)
    G4 = nx.to_undirected(H2)
    lsd = netrd.distance.NetLSD().dist(G3,G4)
    print(lsd)

def lapl(fromdate1,todate1,fromdate2,todate2):
    G1 = controlroom(fromdate1, todate1)
    G2 = controlroom(fromdate2, todate2)
    H1 = loseweight(G1)
    H2 = loseweight(G2)
    G3 = nx.to_undirected(H1)
    G4 = nx.to_undirected(H2)
    lapl = netrd.distance.LaplacianSpectral().dist(G3,G4)
    print(lapl)

def portrait(fromdate1,todate1,fromdate2,todate2):
    G1 = controlroom(fromdate1, todate1)
    G2 = controlroom(fromdate2, todate2)
    H1 = loseweight(G1)
    H2 = loseweight(G2)
    G3 = nx.to_undirected(G1)
    G4 = nx.to_undirected(G2)
    H3 = loseweight(G3)
    H4 = loseweight(G4)
    portrait = netrd.distance.PortraitDivergence().dist(G3,G4)
    portrait2 = netrd.distance.PortraitDivergence().dist(H3,H4)
    print(portrait,portrait2)
    port1 = portrait_divergence_weighted(G3,G4)
    port2 = portrait_divergence(H3,H4)
    port3 = portrait_divergence_weighted(G1,G2)
    port4 = portrait_divergence_weighted(H1,H2)
    print("this be some new undirected shit",port1,port2)
    print("this be some new directed shit",port3,port4)


def example():
    G1 = nx.Graph()
    G2 = nx.Graph()
    edges1 = [("UK", "Denmark"), ("UK", "Germany"), ("UK", "Spain"), ("UK", "Greece")]
    edges2 = [("UK", "Denmark"), ("UK", "Spain"), ("UK", "France"), ("UK", "Germany"), ("Germany", "Greece")]
    G1.add_edges_from(edges1)
    #G2=G1
    G2.add_edges_from(edges2)
    nodes = list(G1.nodes)
    edges = list(G1.edges)
    nodes2 = list(G2.nodes)
    edges2 = list(G2.edges)
    f = open("example.gw", "w")
    print("#header section", "LEDA.GRAPH", "string", "void", "-1", "#nodes section", len(nodes), sep='\n', file=f)
    for i in nodes:
        print("|{" + i + "}]", file=f)
    print("\n", file=f)
    print("#edges section", len(edges), sep="\n", file=f)
    for j in range(len(edges)):
        print(edges[j][0], edges[j][1], "0", "|{}|", sep=" ", file=f)
    f.close()
    f = open("example2.gw", "w")
    print("#header section", "LEDA.GRAPH", "string", "void", "-1", "#nodes section", len(nodes2), sep='\n', file=f)
    for i in nodes2:
        print("|{" + i + "}]", file=f)
    print("\n", file=f)
    print("#edges section", len(edges2), sep="\n", file=f)
    for j in range(len(edges2)):
        print(edges2[j][0], edges2[j][1], "0", "|{}|", sep=" ", file=f)
    f.close()
def test2():
    G1=nx.complete_graph(20)
    G2=nx.path_graph(20)
    G3=nx.barabasi_albert_graph(20,5)
    nodes = list(G1.nodes)
    edges = list(G1.edges)
    nodes2 = list(G2.nodes)
    edges2 = list(G2.edges)
    f = open("clique" +str(len(nodes))+".gw", "w")
    print("#header section", "LEDA.GRAPH", "string", "void", "-1", "#nodes section", len(nodes), sep='\n', file=f)
    for i in nodes:
        print("|{" + str(i) + "}]", file=f)
    print("\n", file=f)
    print("#edges section", len(edges), sep="\n", file=f)
    for j in range(len(edges)):
        print(edges[j][0], edges[j][1], "0", "|{}|", sep=" ", file=f)
    f.close()
    f = open("path" +str(len(nodes))+".gw", "w")
    print("#header section", "LEDA.GRAPH", "string", "void", "-1", "#nodes section", len(nodes2), sep='\n', file=f)
    for i in nodes2:
        print("|{" + str(i) + "}]", file=f)
    print("\n", file=f)
    print("#edges section", len(edges2), sep="\n", file=f)
    for j in range(len(edges2)):
        print(edges2[j][0], edges2[j][1], "0", "|{}|", sep=" ", file=f)

    print(portrait_divergence(G1,G2))
    print(portrait_divergence(G2,G3))
    f.close()

def wecompin():
    G1=controlroom("1/3/2012","30/3/2012")
    G2=nx.powerlaw_cluster_graph(len(list(G1.nodes)),len(list(G1.edges)),0.4)
    port=portrait_divergence(G1,G2)
    print(port)

def pyvistest():
    G=controlroom("1/4/2011","30/4/2011")
    #nt = Network('1000px', '1000px')
    nt = Network(notebook=True)
    nt.from_nx(G)
    #nt.show_buttons(filter=['physics'])
    nt.show_buttons()
    nt.show('nx.html')
    #display(HTML('nx.html'))
    #nt.enable_physics(True)
    #nt.show('mygraph.html')



ignite()
#start=time.time()
#get_trans("29/6/2014","30/6/2014")
#get_unique_nodes("29/6/2014","30/6/2014")
#plotting("29/6/2014","30/7/2014")
#plotting("29/3/2014","3/5/2014")
#a = np.random.randint(0, 5, size=(15, 15))
#D = nx.DiGraph(a)
#nx.draw(D, with_labels=True, node_size=1500, alpha=0.3, arrows=True)
#newplotting("29/6/2014","1/7/2014")
#newplotting("29/9/2014","2/10/2014")
#newplotting("9/2/2016","13/2/2016")
#newplotting("5/5/2015","7/5/2015")
#countryaggr("29/6/2014","1/7/2014")
#newplotting("29/6/2014","30/6/2014")
#getstats("1/4/2014", "1/5/2014")
#getstats("1/3/2012","1/4/2012")
#controlroom("1/1/2012","1/2/2012")
#compar("1/2/2012","1/3/2012","1/2/2013","1/3/2013")
#newcomp("1/4/2012","30/4/2012","1/4/2013","30/4/2013")
#testin()
#controlroom("29/6/2014","30/6/2014")
#weightdist("1/1/2012","31/1/2012","regulated",5)
#weightdist("1/1/2012","31/1/2012","regulated")
#controlroom("29/6/2014","30/7/2014",restriction=5)
#controlroom("29/6/2014","30/6/2014")
#f = open("stats.txt", "w+")
#print('Let us begin', file=f)
#massstats("asdf","ASdf")
#f.close()
#multiple("Asdf","asdf")
#degdist("1/4/2014", "1/5/2014")
#pp = PdfPages('weights.pdf')
#multiple("sdaf","asdf")
#pp.close()
#toleda("April",2013)
#edgeli("July",2013)
#portrait("1/2/2012","28/2/2012","1/4/2013","30/4/2013")
#portrait("1/4/2012","30/4/2012","1/4/2013","30/4/2013")
#testin()
#exatmple()
#test2()
#wecompin()
pyvistest()