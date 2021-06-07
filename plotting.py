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
from networkx.algorithms.community import greedy_modularity_communities,asyn_fluidc,girvan_newman,kernighan_lin_bisection
import time
import pygraphviz as pgv
from plug import newplot
from random import seed
import random
import netcomp as nc
import math
import scipy.stats as stats
import netrd
#from pyensae.languages import r2python
from matplotlib.backends.backend_pdf import PdfPages
import sys
from portrait_divergence import portrait_divergence,portrait_divergence_weighted
from pyvis.network import Network
from IPython.core.display import display, HTML
import xlrd
import os
import csv
from itertools import combinations
import pajek_tools as pajek
import mpmath
import igraph as ig
import seaborn as sns; sns.set()
import pyreadr
import plotly.express as px
from cdlib import algorithms,viz


def onefun(TF):
    if TF==True:
        return 1
    else:
        return 0

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

def controlroom(fromdate,todate,category="None",restriction="None",*args):
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
def newplotting(fromdate,todate,category="None",restriction="None",aggregation="None"):
    G = nx.DiGraph()
    #print("stage1")
    trans = get_trans(fromdate, todate, restriction)
    #print("trans",trans)
    nodes=get_unique_nodes(fromdate, todate,category)
    #print("stage2")
    #nodes=cleannodes(nodes,trans)#αφαιρεί κόμβους που δεν έχουν ακμή(χρειάζομαι connected graph)
    nodesup=[(i[0].upper(),i[1],i[2],i[3],i[4]) for i in nodes]#τα κάνω κεφαλαία
    transup=[(i[0].upper(),i[1].upper(),i[2],i[3]) for i in trans]#τα κάνω κεφαλαία
    #print("stage3")
    transup = cleantrans(transup,nodesup) #backup κατάλοιπο για κόμβους που δεν υπάρχουν στο nodes
    #trans=thecleanest(trans,nodes)
    transup = cleantransloop(transup)
    nodesup = cleannodes(nodesup,transup) #στην περίεργη περίπτωση μεμονωμένου κόμβου που δε μας χρησιμεύει
    #print("these are trans homie:",len(transup),"these dem nodes",len(nodesup))
    #testingfun(nodes,trans)
    #print(nodesup)
    #trans = cleantrans(transup,nodesup) #backup κατάλοιπο για κόμβους που δεν υπάρχουν στο trans
    G=nodify(nodesup,G)#φέρνω τους κόμβους σε καταλλληλη μορφή για το γράφημα
    G=edgify(transup,G)#φέρνω τις ακμές σε κατάλληλη μορφή για το γράφημα
    category = {}
    for node in nodesup:
        category[node[0]] = node[1]
    nx.set_node_attributes(G, category, "category")
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
    return G

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
    sql = f"""select TransferringAccountHolder, AcquiringAccountHolder, NbOfUnits, TransactionType
            from Transactions_New where TransactionDate 
            between '{fromdate}' and '{todate}'"""
    cursor.execute(sql)
    all = cursor.fetchall()
    sql = f"""select TransferringAccountHolder, AcquiringAccountHolder, NbOfUnits, TransactionType
            from Transactions_New
            where ((TransferringAccountHolder is NULL) or (AcquiringAccountHolder is NULL))
            and TransactionDate between '{fromdate}' and '{todate}'"""
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
        addition = f"\nand class.category = '{category}'"
    elif len(category.split(","))==2:
        things=category.split(",")
        addition = f"\nand (class.category = '{things[0]}' or class.category = '{things[1]}')"
    else:
        things=category.split(",")
        addition = f"\nand (class.category = '{things[0]}' or class.category = '{things[1]} or or class.category = '{things[2]}')"

    #print("Addition","l"+addition+"l")
    fromd = cleandate(fromdate)
    tod = cleandate(todate)
    sql = f"""select distinct tran.AcquiringAccountHolder, class.category, class.sector, acc.country, class.registry,size.transNumSize
    from Transactions_New as tran, EUTL_AccountHolders as acc,EUTL_AccHolderClassification as class,EUTL_AccHolderTransactingSize as size
    where tran.AcquiringAccountHolder = acc.holderName
    and acc.rawCode = size.holder
    and acc.rawCode = class.holder{addition}
    and tran.TransactionDate 
    between '{fromd[2]}-{fromd[1]}-{fromd[0]}' and '{tod[2]}-{tod[1]}-{tod[0]}'"""
    #print("stage1.1")
    sql2 = f"""select distinct tran.TransferringAccountHolder, class.category, class.sector, acc.country, class.registry, size.transNumSize
    from Transactions_New as tran, EUTL_AccountHolders as acc,EUTL_AccHolderClassification as class,EUTL_AccHolderTransactingSize as size
    where tran.TransferringAccountHolder = acc.holderName
    and acc.rawCode = size.holder
    and acc.rawCode = class.holder{addition}
    and tran.TransactionDate 
    between '{fromd[2]}-{fromd[1]}-{fromd[0]}' and '{tod[2]}-{tod[1]}-{tod[0]}'"""
    #print("stage1.2")
    #print(sql)
    cursor.execute(sql)
    acq = cursor.fetchall()
    #print("stage1.3")
    cursor.execute(sql2)
    tran = cursor.fetchall()
    #print("stage1.4")
    all=get_unique(acq+tran)
    #print("stage1.5")
    #print("nodes",all)
    return all

def get_trans(fromdate,todate,restriction="None"): #συναλλαγές μεταξύ δυο ημερομηνιών
    if restriction!="None":
        addition= f"and NbOfUnits>{restriction}"
    else: addition=""
    fromd=cleandate(fromdate)
    tod=cleandate(todate)
    key = 2 #1 for everything in, 2 for excluding NULL,
    if key ==1 :
        sql = f"""select TransferringAccountHolder, AcquiringAccountHolder, NbOfUnits, TransactionType
        from Transactions_New where
        TransactionDate 
        between '{fromd[2]}-{fromd[1]}-{fromd[0]}' and '{tod[2]}-{tod[1]}-{tod[0]}'
        {addition}"""
    #If we want to exclude NULL appearances do the following
    elif key==2 :
        sql = f"""select TransferringAccountHolder, AcquiringAccountHolder, NbOfUnits, TransactionType
        from Transactions_New
        where ((TransferringAccountHolder is not NULL) and (AcquiringAccountHolder is not NULL))
        and TransactionDate between '{fromd[2]}-{fromd[1]}-{fromd[0]}' and '{tod[2]}-{tod[1]}-{tod[0]}'
        {addition}"""
    cursor.execute(sql)
    tran=cursor.fetchall()
    return tran

def get_trans_country(fromdate,todate):
    fromd = cleandate(fromdate)
    tod = cleandate(todate)
    key = 2  # 1 for everything in, 2 for excluding NULL,
    if key == 1:
        sql = f"""select TransferringRegistry,AcquiringRegistry, NbOfUnits, TransactionType
            from Transactions_New where TransactionDate 
            between '{fromd[2]}-{fromd[1]}-{fromd[0]}' and '{tod[2]}-{tod[1]}-{tod[0]}'"""
    # If we want to exclude NULL appearances do the following
    elif key == 2:
        sql = f"""select TransferringRegistry,AcquiringRegistry, NbOfUnits, TransactionType
            from Transactions_New
            where ((TransferringAccountHolder is not NULL) and (AcquiringAccountHolder is not NULL))
            and TransactionDate between '{fromd[2]}-{fromd[1]}-{fromd[0]}' and '{tod[2]}-{tod[1]}-{tod[0]}'"""
    cursor.execute(sql)
    tran = cursor.fetchall()
    return tran

def graphify(dic):
    df = pd.DataFrame(dic)
    G = nx.from_pandas_edgelist(df, 'from', 'to', create_using=nx.DiGraph())
    nx.draw(G, with_labels=False, node_size=1500, alpha=0.3, arrows=True)

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
    sql0=f"""select tran.TransferringAccountHolder, class.category, class.sector,acc.country, class.registry
        from Transactions_New as tran, EUTL_AccountHolders as acc,EUTL_AccHolderClassification as class
        where tran.TransferringAccountHolder = acc.holderName
        and acc.rawCode = class.holder
        and tran.TransactionDate 
        between '{fromd[2]}-{fromd[1]}-{fromd[0]}' and '{tod[2]}-{tod[1]}-{tod[0]}'"""
    sql1 = f"""select tran.AcquiringAccountHolder, class.category, class.sector,acc.country, class.registry
       from Transactions_New as tran, EUTL_AccountHolders as acc,EUTL_AccHolderClassification as class
       where tran.AcquiringAccountHolder = acc.holderName
       and acc.rawCode = class.holder
       and tran.TransactionDate 
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

def getstats(fromdate,todate,category="None"):
    G = controlroom(fromdate,todate,category=category)
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
    #print(list(nx.isolates(G)))
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

def massstats(fromdate,todate,category="None"):
    months=["January","February","March","April","May","June","July","August","September","October","November","December"]
    F=T=["1","1","2005"]
    dates=[("1/1/","31/1/"),("1/2/","28/2/"),("1/3/","31/3/"),("1/4/","30/4/"),("1/5/","31/5/"),("1/6/","30/6/"),("1/7/","31/7/"),("1/8/","31/8/")
        ,("1/9/","30/9/"),("1/10/","31/10/"),("1/11/","30/11/"),("1/12/","31/12/")]
    for i in range(10):
        year=2006+i
        print(year,file=f)
        for j in range(12):
            print(months[j],file=f)
            getstats(dates[j][0]+str(year),dates[j][1]+str(year),category=category)

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


def countdiffrences():
    df = pd.read_csv("directedmonthscomparison.csv", sep=';')
    print(len(df))
    count = [0, 0, 0, 0, 0, 0, 0]
    countall = 0
    for row in df.itertuples():
        for i in row:
            if type(i) == str or type(i) == int:
                continue
            countall += 1
            if i <= 0.4:
                count[0] += 1
            elif i <= 0.5:
                count[1] += 1
            elif i <= 0.6:
                count[2] += 1
            elif i <= 0.7:
                count[3] += 1
            elif i <= 0.8:
                count[4] += 1
            elif i <= 0.9:
                count[5] += 1
            else:
                count[6] += 1

    for i in range(7):
        print(f"count{i + 4}0", count[i] * 100 / countall)
    df = pd.read_csv("monthscomparison.csv", sep=',')
    print(len(df))
    count = [0, 0, 0, 0, 0, 0, 0]
    countall = 0
    for row in df.itertuples():
        for i in row:
            if type(i) == str or type(i) == int:
                continue
            countall += 1
            if i <= 0.4:
                count[0] += 1
            elif i <= 0.5:
                count[1] += 1
            elif i <= 0.6:
                count[2] += 1
            elif i <= 0.7:
                count[3] += 1
            elif i <= 0.8:
                count[4] += 1
            elif i <= 0.9:
                count[5] += 1
            else:
                count[6] += 1

    for i in range(7):
        print(f"count{i + 4}0", count[i] * 100 / countall)

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
    #print("minas",minas)
    #print(dates[minas][0]+str(year),dates[minas][1]+str(year))
    G=controlroom(dates[minas][0]+str(year),dates[minas][1]+str(year))
    nodes=list(G.nodes())
    nodesindex=[nodes.index(node) for node in nodes]
    #print(nodesindex)
    edges=list(G.edges)
    edgesindex=[]
    for edge in edges:
        edgesindex.append((nodes.index(edge[0]),nodes.index(edge[1])))
    #print(edgesindex)
    #print("wegotthisfar")
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
    #print("minas", minas)
    #print(dates[minas][0] + str(year), dates[minas][1] + str(year))
    G = controlroom(dates[minas][0] + str(year), dates[minas][1] + str(year))
    edges=list(G.edges)
    f = open(f"edgelist-{month}{year}.gw", "w")
    #for edge in edges:
        #print(edge[0],edge[1],file=f)
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
    #G1=controlroom("1/3/2012","30/3/2012")
    #G2=nx.barabasi_albert_graph(len(list(G1.nodes)),1)
    G1=nx.path_graph()
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

def visual(month,year,category="None"):
    months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
              "November", "December"]
    dates = [("1/1/", "31/1/"), ("1/2/", "28/2/"), ("1/3/", "31/3/"), ("1/4/", "30/4/"), ("1/5/", "31/5/"),
             ("1/6/", "30/6/"), ("1/7/", "31/7/"), ("1/8/", "31/8/")
        , ("1/9/", "30/9/"), ("1/10/", "31/10/"), ("1/11/", "30/11/"), ("1/12/", "31/12/")]
    minas = months.index(month)
    G = controlroom(dates[minas][0] + str(year), dates[minas][1] + str(year),category=category)
    nt = Network(notebook=True)
    nt.from_nx(G)
    nt.show_buttons()
    nt.show(f"{month}{year}.html")

def doingrand():
    #G1=controlroom("1/3/2012","30/3/2012")
    G=nx.watts_strogatz_graph(200,2,0.1)
    nt = Network(notebook=True)
    nt.from_nx(G)
    nt.show_buttons()
    nt.show('watts.html')
    G1=nx.newman_watts_strogatz_graph(200,2,0.1)
    nt1 = Network(notebook=True)
    nt1.from_nx(G1)
    nt1.show_buttons()
    nt1.show('newman.html')

def sotiris():
    months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
              "November", "December"]
    for i in range(3):
        for j in months:
            toleda(j,2010+i)

def manytests():
    lis=[10,20]
    graphs=[]
    for i in lis:
        graphs.append(nx.path_graph(i))
        graphs.append(nx.cycle_graph(i))
        graphs.append(nx.complete_graph(i))
        graphs.append(nx.grid_graph((i,i)))
        graphs.append(nx.star_graph(i))
        graphs.append(nx.ladder_graph(i))
        graphs.append(nx.wheel_graph(i))
    f= open('portraittrials.txt','w+')
    indices=["path","cycle","clique","grid","star","ladder","wheel"]
    strs=[""]*len(graphs)
    print(strs)
    counter=0
    for i in range(len(strs)):
        print(counter//len(indices))
        print("counter",counter)
        strs[i]=str(indices[i%len(indices)]) + str(lis[counter//len(indices)]) + "\t"
        counter+=1
    print(strs)
    for i in range(len(graphs)):
        for j in range(i):
            print(i,j)
            strs[i]= strs[i] + "\t" + str(format(portrait_divergence(graphs[i],graphs[j]),'.4f')) + "\t"
    for i in strs:
        print(i, file=f)
    f.close()

def manytests2():
    lis = [10, 20, 50, 100]
    graphs = []
    for i in lis:
        graphs.append(nx.path_graph(i))
        graphs.append(nx.cycle_graph(i))
        graphs.append(nx.complete_graph(i))
        graphs.append(nx.grid_graph((i, i)))
        graphs.append(nx.star_graph(i))
        graphs.append(nx.ladder_graph(i))
        graphs.append(nx.wheel_graph(i))
    indices = ["path", "cycle", "clique", "grid", "star", "ladder", "wheel"]
    strs = [""] * len(graphs)
    #print(strs)
    counter = 0
    #for i in range(len(strs)):
    #    print(counter // len(indices))
    #    print("counter", counter)
    #    strs[i] = str(indices[i % len(indices)]) + str(lis[counter // len(indices)]) + " "
    #    counter += 1
    #print(strs)
    for i in range(len(graphs)):
        for j in range(i+1):
            print(i,j)
            strs[i]= strs[i] + " " + str(format(portrait_divergence(graphs[i],graphs[j]),'.4f')) + " "
    empt=[]
    colum=[]
    counter = 0
    for i in range(len(graphs)):
        colum.append(str(indices[i % len(indices)]) + str(lis[counter // len(indices)]))
        counter += 1
    for i in strs:
        empt.append(i.split())
    #for i in empt:
    data={}
    for i in range(len(colum)):
        data[colum[i]]=[0.0]*len(colum)
        for j in range(i+1):
            data[colum[i]][j]=float(empt[i][j])
    #for i in range(len(colum)):
    #    for j in range(i+1):
    #        data[colum[j]][i]=empt[j][i]
    #mat = np.matrix(empt)
    #df = pd.DataFrame(data=empt,columns=colum)
    df=pd.DataFrame(data)
    #print("df",df)
    #df.to_csv('outfile.csv')
    #with open('portraittests.txt', 'wb') as f:
    #    for line in mat:
    #        np.savetxt(f, line)
    #np.savetxt(r'/Users/user/PycharmProjects/secondtry/outwithit.txt', df.values,fmt='%1.3f')
    nup=df.to_numpy()
    #print("nup",nup)
    for i in range(len(nup)):
        for j in range(i):
            nup[i][j]=nup[j][i]
    #print(nup)
    f=open("portrait_stats.csv","w+")
    col=["columns"]+colum
    final=[]
    final.append(', '.join(col))
    for i in range(len(nup)):
        final.append(colum[i]+", "+', '.join(map(str,nup[i])))
    for i in final:
        print(i,file=f)
    f.close()

def doit():
    a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    mat = np.matrix(a)
    with open('outfile.txt', 'wb') as f:
        for line in mat:
            np.savetxt(f, line, fmt='%.2f')

def tempgraph():
    #G = nx.grid_graph((50, 50))
    G= nx.wheel_graph(50)
    print(nx.is_directed(G))
    nt = Network(notebook=True)
    nt.from_nx(G)
    nt.show_buttons()
    nt.show('wheel.html')
ignite()

def reggovfin(month,year):
    months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
              "November", "December"]
    dates = [("1/1/", "31/1/"), ("1/2/", "28/2/"), ("1/3/", "31/3/"), ("1/4/", "30/4/"), ("1/5/", "31/5/"),
             ("1/6/", "30/6/"), ("1/7/", "31/7/"), ("1/8/", "31/8/")
        , ("1/9/", "30/9/"), ("1/10/", "31/10/"), ("1/11/", "30/11/"), ("1/12/", "31/12/")]
    minas = months.index(month)
    G = controlroom(dates[minas][0] + str(year), dates[minas][1] + str(year))
    print("all edges",len(list(G.edges)))
    all=len(list(G.edges))
    G1 = controlroom(dates[minas][0] + str(year), dates[minas][1] + str(year),"regulated")
    regedges = len(list(G1.edges))
    print("between regulated",regedges,f'{regedges/all*100:.3g}%')
    G2 = controlroom(dates[minas][0] + str(year), dates[minas][1] + str(year), "governmental")
    govedges = len(list(G2.edges))
    print("between governmental",govedges,f'{govedges/all*100:.3g}%')
    G3 = controlroom(dates[minas][0] + str(year), dates[minas][1] + str(year), "financial")
    finedges= len(list(G3.edges))
    print("between financial",finedges,f'{finedges/all*100:.3g}%')
    G4 = controlroom(dates[minas][0] + str(year), dates[minas][1] + str(year), "regulated,financial")
    regfin =len(list(G4.edges))- len(list(G1.edges))-len(list(G3.edges))
    print("between regulated and financial",regfin,f'{regfin/all*100:.3g}%')
    G5 = controlroom(dates[minas][0] + str(year), dates[minas][1] + str(year), "regulated,governmental")
    reggov = len(list(G5.edges)) - len(list(G1.edges)) - len(list(G2.edges))
    print("between regulated and governmental", reggov,f'{reggov/all*100:.3g}%')
    G6 = controlroom(dates[minas][0] + str(year), dates[minas][1] + str(year), "governmental,financial")
    fingov = len(list(G6.edges)) - len(list(G2.edges)) - len(list(G3.edges))
    print("between governmental and financial", fingov,f'{fingov/all*100:.3g}%')

def reggovfinfile(month,year,f):
    months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
              "November", "December"]
    dates = [("1/1/", "31/1/"), ("1/2/", "28/2/"), ("1/3/", "31/3/"), ("1/4/", "30/4/"), ("1/5/", "31/5/"),
             ("1/6/", "30/6/"), ("1/7/", "31/7/"), ("1/8/", "31/8/")
        , ("1/9/", "30/9/"), ("1/10/", "31/10/"), ("1/11/", "30/11/"), ("1/12/", "31/12/")]
    minas = months.index(month)
    G = controlroom(dates[minas][0] + str(year), dates[minas][1] + str(year))
    print("all edges",len(list(G.edges)),file=f)
    all=len(list(G.edges))
    G1 = controlroom(dates[minas][0] + str(year), dates[minas][1] + str(year),"regulated")
    regedges = len(list(G1.edges))
    print("between regulated",regedges,f'{regedges/all*100:.3g}%',file=f)
    G2 = controlroom(dates[minas][0] + str(year), dates[minas][1] + str(year), "governmental")
    govedges = len(list(G2.edges))
    print("between governmental",govedges,f'{govedges/all*100:.3g}%',file=f)
    G3 = controlroom(dates[minas][0] + str(year), dates[minas][1] + str(year), "financial")
    finedges= len(list(G3.edges))
    print("between financial",finedges,f'{finedges/all*100:.3g}%',file=f)
    G4 = controlroom(dates[minas][0] + str(year), dates[minas][1] + str(year), "regulated,financial")
    regfin =len(list(G4.edges))- len(list(G1.edges))-len(list(G3.edges))
    print("between regulated and financial",regfin,f'{regfin/all*100:.3g}%',file=f)
    G5 = controlroom(dates[minas][0] + str(year), dates[minas][1] + str(year), "regulated,governmental")
    reggov = len(list(G5.edges)) - len(list(G1.edges)) - len(list(G2.edges))
    print("between regulated and governmental", reggov,f'{reggov/all*100:.3g}%',file=f)
    G6 = controlroom(dates[minas][0] + str(year), dates[minas][1] + str(year), "governmental,financial")
    fingov = len(list(G6.edges)) - len(list(G2.edges)) - len(list(G3.edges))
    print("between governmental and financial", fingov,f'{fingov/all*100:.3g}%',file=f)

def trials(month,year):
    months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
              "November", "December"]
    dates = [("1/1/", "31/1/"), ("1/2/", "28/2/"), ("1/3/", "31/3/"), ("1/4/", "30/4/"), ("1/5/", "31/5/"),
             ("1/6/", "30/6/"), ("1/7/", "31/7/"), ("1/8/", "31/8/")
        , ("1/9/", "30/9/"), ("1/10/", "31/10/"), ("1/11/", "30/11/"), ("1/12/", "31/12/")]
    minas = months.index(month)
    lis = [10, 20, 50, 100]
    graphs = []
    G = controlroom(dates[minas][0] + str(year), dates[minas][1] + str(year))
    for i in lis:
        graphs.append(nx.path_graph(i))
        graphs.append(nx.cycle_graph(i))
        graphs.append(nx.complete_graph(i))
        graphs.append(nx.grid_graph((i, i)))
        graphs.append(nx.star_graph(i))
        graphs.append(nx.ladder_graph(i))
        graphs.append(nx.wheel_graph(i))
    indices = ["path", "cycle", "clique", "grid", "star", "ladder", "wheel"]
    strs = [""] * len(graphs)
    for i in range(len(graphs)):
        print(indices[i%len(indices)], portrait_divergence(G,graphs[i]))

def vol(month,year):
    months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
              "November", "December"]
    dates = [("1/1/", "31/1/"), ("1/2/", "28/2/"), ("1/3/", "31/3/"), ("1/4/", "30/4/"), ("1/5/", "31/5/"),
             ("1/6/", "30/6/"), ("1/7/", "31/7/"), ("1/8/", "31/8/")
        , ("1/9/", "30/9/"), ("1/10/", "31/10/"), ("1/11/", "30/11/"), ("1/12/", "31/12/")]
    minas = months.index(month)
    #G = controlroom(dates[minas][0] + str(year), dates[minas][1] + str(year))
    #edges=list(G.edges())
    #count=0
    #for edge in edges:
    #    count+=G.get_edge_data(edge[0],edge[1])["weight"]
    G = controlroom(dates[minas][0] + str(year), dates[minas][1] + str(year))
    all = list(G.edges())
    weightall = 0
    for edge in all:
        weightall += G.get_edge_data(edge[0], edge[1])["weight"]
    print("weightall",format(weightall, ","))
    G1 = controlroom(dates[minas][0] + str(year), dates[minas][1] + str(year), "regulated")
    reg = list(G1.edges)
    weightreg = 0
    for edge in reg:
        weightreg += G1.get_edge_data(edge[0], edge[1])["weight"]
    print("weightreg", format(weightreg, ","))
    G2 = controlroom(dates[minas][0] + str(year), dates[minas][1] + str(year), "governmental")
    gov = list(G2.edges)
    weightgov = 0
    for edge in gov:
        weightgov += G2.get_edge_data(edge[0], edge[1])["weight"]
    print("weightgov", format(weightgov, ","))
    G3 = controlroom(dates[minas][0] + str(year), dates[minas][1] + str(year), "financial")
    fin = list(G3.edges)
    weightfin = 0
    for edge in fin:
        weightfin += G3.get_edge_data(edge[0], edge[1])["weight"]
    print("weightfin", format(weightfin, ","))
    G4 = controlroom(dates[minas][0] + str(year), dates[minas][1] + str(year), "regulated,financial")
    regfin = list(filter(lambda x: x not in reg+fin, list(G4.edges)))
    weightregfin = 0
    for edge in regfin:
        weightregfin += G4.get_edge_data(edge[0], edge[1])["weight"]
    print("weightregfin", format(weightregfin, ","))
    G5 = controlroom(dates[minas][0] + str(year), dates[minas][1] + str(year), "regulated,governmental")
    reggov = list(filter(lambda x: x not in reg + gov, list(G5.edges)))
    weightreggov = 0
    for edge in reggov:
        weightreggov += G5.get_edge_data(edge[0], edge[1])["weight"]
    print("weightreggov", format(weightreggov, ","))
    G6 = controlroom(dates[minas][0] + str(year), dates[minas][1] + str(year), "governmental,financial")
    fingov = list(filter(lambda x: x not in reg + gov, list(G6.edges)))
    weightfingov = 0
    for edge in fingov:
        weightfingov += G6.get_edge_data(edge[0], edge[1])["weight"]
    print("weightfingov", format(weightfingov, ","))

def vol2(month,year,f):
    months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
              "November", "December"]
    dates = [("1/1/", "31/1/"), ("1/2/", "28/2/"), ("1/3/", "31/3/"), ("1/4/", "30/4/"), ("1/5/", "31/5/"),
             ("1/6/", "30/6/"), ("1/7/", "31/7/"), ("1/8/", "31/8/")
        , ("1/9/", "30/9/"), ("1/10/", "31/10/"), ("1/11/", "30/11/"), ("1/12/", "31/12/")]
    minas = months.index(month)
    #G = controlroom(dates[minas][0] + str(year), dates[minas][1] + str(year))
    #edges=list(G.edges())
    #count=0
    #for edge in edges:
    #    count+=G.get_edge_data(edge[0],edge[1])["weight"]
    G = controlroom(dates[minas][0] + str(year), dates[minas][1] + str(year))
    all = list(G.edges())
    weightall = 0
    for edge in all:
        weightall += G.get_edge_data(edge[0], edge[1])["weight"]
    print("weightall",format(weightall, ","),file=f)
    G1 = controlroom(dates[minas][0] + str(year), dates[minas][1] + str(year), "regulated")
    reg = list(G1.edges)
    weightreg = 0
    for edge in reg:
        weightreg += G1.get_edge_data(edge[0], edge[1])["weight"]
    print("weightreg", format(weightreg, ","),file=f)
    G2 = controlroom(dates[minas][0] + str(year), dates[minas][1] + str(year), "governmental")
    gov = list(G2.edges)
    weightgov = 0
    for edge in gov:
        weightgov += G2.get_edge_data(edge[0], edge[1])["weight"]
    print("weightgov", format(weightgov, ","),file=f)
    G3 = controlroom(dates[minas][0] + str(year), dates[minas][1] + str(year), "financial")
    fin = list(G3.edges)
    weightfin = 0
    for edge in fin:
        weightfin += G3.get_edge_data(edge[0], edge[1])["weight"]
    print("weightfin", format(weightfin, ","),file=f)
    G4 = controlroom(dates[minas][0] + str(year), dates[minas][1] + str(year), "regulated,financial")
    regfin = list(filter(lambda x: x not in reg+fin, list(G4.edges)))
    weightregfin = 0
    for edge in regfin:
        weightregfin += G4.get_edge_data(edge[0], edge[1])["weight"]
    print("weightregfin", format(weightregfin, ","),file=f)
    G5 = controlroom(dates[minas][0] + str(year), dates[minas][1] + str(year), "regulated,governmental")
    reggov = list(filter(lambda x: x not in reg + gov, list(G5.edges)))
    weightreggov = 0
    for edge in reggov:
        weightreggov += G5.get_edge_data(edge[0], edge[1])["weight"]
    print("weightreggov", format(weightreggov, ","),file=f)
    G6 = controlroom(dates[minas][0] + str(year), dates[minas][1] + str(year), "governmental,financial")
    fingov = list(filter(lambda x: x not in reg + gov, list(G6.edges)))
    weightfingov = 0
    for edge in fingov:
        weightfingov += G6.get_edge_data(edge[0], edge[1])["weight"]
    print("weightfingov", format(weightfingov, ","),file=f)

def nightly():
    months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
              "November", "December"]
    f= open("nightlyfiles.txt","w+")
    for i in range(2006,2016):
        for month in months:
            print(month,i,file=f)
            reggovfinfile(month,i,f)
            print("weights",file=f)
            vol2(month,i,f)
    f.close()

def runtrials():
    #G=controlroom("1/1/2011","31/1/2011")
    G=nx.cycle_graph(20)
    G1=nx.path_graph(20)
    G2=nx.ladder_graph(20)
    nx.draw(G2)
    nt = Network(notebook=True)
    nt.from_nx(G2)
    # nt.show_buttons(filter=['physics'])
    nt.show_buttons()
    nt.show('laddergraph.html')
    print(portrait_divergence(G,G1))

def whatamidoing():
    G=controlroom("1/9/2013","30/9/2013")
    G1=controlroom("1/9/2012","30/9/2012")
    while len(list(G.edges))<len(list(G1.edges)):
        randedge = random.choice(list(G1.edges))
        #print("compar edges",len(list(G.edges)),len(list(G1.edges)))
        G1.remove_edge(randedge[0],randedge[1])
    print("compar edges",len(list(G.edges)),len(list(G1.edges)))
    print(portrait_divergence(G,G1))

def centric():
    G = controlroom("1/1/05", "31/12/05")
    gedges = list(G.edges())
    gnodes = list(G.nodes())
    print("edges",len(gedges),"nodes",len(gnodes))
    prob1=len(gnodes)/(len(gnodes)+len(gnodes)/2)
    prob2=len(gnodes)/(2*(len(gnodes)+len(gnodes)/2))

    network=nx.Graph()
    network.add_nodes_from(gnodes)
    potedges = list(combinations(network.nodes, 2))
    timestamp1=time.time()-start
    print("timestamp1",timestamp1)
    for potential in potedges:
        rand = random.uniform(0,1)
        if potential in gedges:
            if rand > prob1:
                network.add_edge(potential[0],potential[1])
        else:
            if rand > prob2:
                network.add_edge(potential[0],potential[1])
    #       newplot(network)
    nt = Network(notebook=True)
    nt.from_nx(network)
    # nt.show_buttons(filter=['physics'])
    nt.show_buttons()
    nt.show('centrictest.html')
    endtime=time.time()-timestamp1
    print("ending",endtime)
    print(prob1,prob2)


def centricbetter():
    G = controlroom("1/1/05", "31/12/05")
    print("density baby",nx.density(G))
    gedges = list(G.edges)
    gnodes = list(G.nodes)
    print("edges", len(gedges), "nodes", len(gnodes))
    prob1 = len(gnodes) / (len(gnodes) + len(gnodes) / 2)
    prob2 = len(gnodes) / (2 * (len(gnodes) + len(gnodes) / 2))
    timestamp1= time.time()-start
    print("timestamp1",timestamp1)
    network = nx.Graph()
    network.add_nodes_from(gnodes)
    potedges = list(combinations(network.nodes, 2))
    print(len(potedges))
    timestamp2= time.time() - timestamp1
    print("timestamp2",timestamp2)
    #nonedges=[x for x in potedges if x not in gedges]
    nonedges=set(potedges) - set(gedges)
    print(len(gedges),len(nonedges),len(potedges))
    print("finishing time", time.time() - timestamp2)
    counter=0
    print("HOW DID I GET HERE")
    for i in gedges:
        counter+=1
        #print("edges",counter)
        rand = random.uniform(0,1)
        if rand > prob1:
            network.add_edge(i[0], i[1])
    for i in nonedges:
        counter += 1
        #print("nonedges", counter)
        rand = random.uniform(0,1)
        if rand > prob2:
            network.add_edge(i[0], i[1])
    """nt = Network(notebook=True)
    print("are we here")
    nt.from_nx(network)
    print("or are we here")
    # nt.show_buttons(filter=['physics'])
    nt.show_buttons()
    print("the fuck is happenin")
    nt.show('centrictest.html')"""
    #print("difference",portrait_divergence(network,G))
    endtime = time.time() - timestamp2
    print("ending", endtime)

def getshitfast(fromdate, todate, category="None", restriction="None", aggregation="None"):
    G = nx.DiGraph()
    # print("stage1")
    trans = get_trans(fromdate, todate, restriction)
    # print("trans",trans)
    nodes = get_unique_nodes(fromdate, todate, category)
    # print("stage2")
    # nodes=cleannodes(nodes,trans)#αφαιρεί κόμβους που δεν έχουν ακμή(χρειάζομαι connected graph)
    nodesup = [(i[0].upper(), i[1], i[2], i[3], i[4]) for i in nodes]  # τα κάνω κεφαλαία
    transup = [(i[0].upper(), i[1].upper(), i[2], i[3]) for i in trans]  # τα κάνω κεφαλαία
    # print("stage3")
    transup = cleantrans(transup, nodesup)  # backup κατάλοιπο για κόμβους που δεν υπάρχουν στο nodes
    # trans=thecleanest(trans,nodes)
    transup = cleantransloop(transup)
    nodesup = cleannodes(nodesup, transup)  # στην περίεργη περίπτωση μεμονωμένου κόμβου που δε μας χρησιμεύει
    # print("these are trans homie:",len(transup),"these dem nodes",len(nodesup))
    # testingfun(nodes,trans)
    # print(nodesup)
    # trans = cleantrans(transup,nodesup) #backup κατάλοιπο για κόμβους που δεν υπάρχουν στο trans
    print("nodesup",len(nodesup))
    print("transup",len(transup))

    #G = nodify(nodesup, G)  # φέρνω τους κόμβους σε καταλλληλη μορφή για το γράφημα
    #G = edgify(transup, G)  # φέρνω τις ακμές σε κατάλληλη μορφή για το γράφημα
    # newplot(G)#κάλεσμα συνάρτησης από το plug.py για το σχεδιασμό
    # nx.draw(G, with_labels=True, node_size=1500, alpha=0.3, arrows=True)
    # plt.show()
    # dg.plot_2d()
    return G

def newfaststats(fromdate, todate,
                     category="None"):  # return list of unique account holders [('node1,'category',
    if category == "None":
        addition = ""
    elif "," not in category:
        addition = f"\nand class.category = '{category}'"
    elif len(category.split(",")) == 2:
        things = category.split(",")
        addition = f"\nand (class.category = '{things[0]}' or class.category = '{things[1]}')"
    else:
        things = category.split(",")
        addition = f"\nand (class.category = '{things[0]}' or class.category = '{things[1]} or or class.category = '{things[2]}')"

    # print("Addition","l"+addition+"l")
    fromd = cleandate(fromdate)
    tod = cleandate(todate)
    sql = f"""select distinct tran.AcquiringAccountHolder, class.category, class.sector, acc.country, class.registry,size.avgTransNum
    from Transactions_New as tran, EUTL_AccountHolders as acc,EUTL_AccHolderClassification as class, EUTL_AccHolderTransactingSize as size
    where tran.AcquiringAccountHolder = acc.holderName
    and acc.rawCode = size.holder
    and acc.rawCode = class.holder{addition}
    and tran.TransactionDate 
    between '{fromd[2]}-{fromd[1]}-{fromd[0]}' and '{tod[2]}-{tod[1]}-{tod[0]}'"""
    # print("stage1.1")
    sql2 = f"""select distinct tran.TransferringAccountHolder, class.category, class.sector, acc.country, class.registry,size.avgTransNum
    from Transactions_New as tran, EUTL_AccountHolders as acc,EUTL_AccHolderClassification as class, EUTL_AccHolderTransactingSize as size
    where tran.TransferringAccountHolder = acc.holderName
    and acc.rawCode = size.holder
    and acc.rawCode = class.holder{addition}
    and tran.TransactionDate 
    between '{fromd[2]}-{fromd[1]}-{fromd[0]}' and '{tod[2]}-{tod[1]}-{tod[0]}'"""
    # print("stage1.2")
    # print(sql)
    cursor.execute(sql)
    acq = cursor.fetchall()
    # print("stage1.3")
    cursor.execute(sql2)
    tran = cursor.fetchall()
    # print("stage1.4")
    all = get_unique(acq + tran)
    # print("stage1.5")
    # print("nodes",all)
    return all

def damnindians(month,year,category="None"):
    months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
              "November", "December"]
    dates = [("1/1/", "31/1/"), ("1/2/", "28/2/"), ("1/3/", "31/3/"), ("1/4/", "30/4/"), ("1/5/", "31/5/"),
             ("1/6/", "30/6/"), ("1/7/", "31/7/"), ("1/8/", "31/8/")
        , ("1/9/", "30/9/"), ("1/10/", "31/10/"), ("1/11/", "30/11/"), ("1/12/", "31/12/")]
    f=open(f"{month}{year}.paj",'w+')
    g=open(f"{month}{year}sup.txt",'w+')
    minas = months.index(month)
    G,finalnodes = slovenian(month,year,category=category)
    #G=controlroom("1/1/2011","5/1/2011")
    nodes=list(G.nodes())
    edges=list(G.edges())
    #print(f"""*Network {month}{year}""",file=f)
    print("*Vertices",len(nodes),file=f)
    edgesindex = []
    for edge in edges:
        edgesindex.append((nodes.index(edge[0])+1, nodes.index(edge[1])+1))#τα αυξάνω κατά 1 γιατί θα τυπώσω τους κόμβους με +1 μετάθεση
    nodesindex = [nodes.index(node) for node in nodes]
    for i in range(len(nodesindex)):
        print(nodesindex[i]+1,"\""+nodes[i]+"\"",sep=" ",file=f)

        #print(nodesindex[i]+1, sep="\t", file=f)
    #print("*Arcs",file=f)
    print("*Edges",file=f)
    for j in range(len(edgesindex)):
        print(edgesindex[j][0],edgesindex[j][1],sep=" ",file = f)
    print(finalnodes)
    for node in finalnodes:
        print(node[5],file=g)
    g.close()
    f.close()


def slovenian(month,year,category="None",restriction="None"):
    G=nx.Graph()
    months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
              "November", "December"]
    dates = [("1/1/", "31/1/"), ("1/2/", "28/2/"), ("1/3/", "31/3/"), ("1/4/", "30/4/"), ("1/5/", "31/5/"),
             ("1/6/", "30/6/"), ("1/7/", "31/7/"), ("1/8/", "31/8/")
        , ("1/9/", "30/9/"), ("1/10/", "31/10/"), ("1/11/", "30/11/"), ("1/12/", "31/12/")]
    minas = months.index(month)
    nodes=newfaststats(dates[minas][0] + str(year), dates[minas][1] + str(year), category=category)


    #print(nodes)
    trans = get_trans(dates[minas][0] + str(year), dates[minas][1] + str(year), restriction=restriction)
    nodesup = [(i[0].upper(), i[1], i[2], i[3], i[4],i[5]) for i in nodes]  # τα κάνω κεφαλαία
    transup = [(i[0].upper(), i[1].upper(), i[2], i[3]) for i in trans]  # τα κάνω κεφαλαία
    # print("stage3")
    transup = cleantrans(transup, nodesup)  # backup κατάλοιπο για κόμβους που δεν υπάρχουν στο nodes
    # trans=thecleanest(trans,nodes)
    transup = cleantransloop(transup)
    nodesup = cleannodes(nodesup, transup)  # στην περίεργη περίπτωση μεμονωμένου κόμβου που δε μας χρησιμεύει
    #print("nodes after cleaning",nodes)
    G = nodify(nodesup, G)  # φέρνω τους κόμβους σε καταλλληλη μορφή για το γράφημα
    G = edgify(transup, G)  # φέρνω τις ακμές σε κατάλληλη μορφή για το γράφημα
    category = {}
    for node in nodesup:
        category[node[0]] = node[1]
    nx.set_node_attributes(G, category, "category")
    # newplot(G)#κάλεσμα συνάρτησης από το plug.py για το σχεδιασμό
    # nx.draw(G, with_labels=True, node_size=1500, alpha=0.3, arrows=True)
    # plt.show()
    # dg.plot_2d()
    #print(nodesup,list(G.nodes()))
    return (G,nodesup)
    #G = controlroom(dates[minas][0] + str(year), dates[minas][1] + str(year), category=category)

def somepypistuff(month,year,category="None"):
    months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
              "November", "December"]
    dates = [("1/1/", "31/1/"), ("1/2/", "28/2/"), ("1/3/", "31/3/"), ("1/4/", "30/4/"), ("1/5/", "31/5/"),
             ("1/6/", "30/6/"), ("1/7/", "31/7/"), ("1/8/", "31/8/")
        , ("1/9/", "30/9/"), ("1/10/", "31/10/"), ("1/11/", "30/11/"), ("1/12/", "31/12/")]
    f = open(f"""{month}{year}.paj""", 'w+')
    minas = months.index(month)
    G = controlroom(dates[minas][0] + str(year), dates[minas][1] + str(year), category=category)
    edges= list(G.edges())
    nodes= list(G.nodes())
    df= pd.DataFrame
    l=[]
    for edge in edges:
        l.append([edge[0],edge[1]])
    #data=np.array(l)
    df =pd.DataFrame(l,columns=["transfering","acquiring"])
    writer = pajek.PajekWriter(df,directed=False,citing_colname="transfering",cited_colname="acquiring")
    writer.write("output.net")

def altern(month,year,category="None"):
    G,finalnodes = slovenian(month,year,category=category)
    nx.write_pajek(G, f"dir/{month}{year}.net")
    g=open(f"dir/{month}{year}sup.txt",'w+')
    stri=""
    #for node in finalnodes:
    #    stri=str(G.degree[node[0]]) +" "
        #print(G.degree[node[0]],file=g)
    for node in finalnodes:
        print(node[5],G.degree(node[0]),node[3],node[1],file=g)
    #print(stri,file=g)
    g.close()

def custompaj(month,year,category="None",restriction="None"):
    months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
              "November", "December"]
    dates = [("1/1/", "31/1/"), ("1/2/", "28/2/"), ("1/3/", "31/3/"), ("1/4/", "30/4/"), ("1/5/", "31/5/"),
             ("1/6/", "30/6/"), ("1/7/", "31/7/"), ("1/8/", "31/8/")
        , ("1/9/", "30/9/"), ("1/10/", "31/10/"), ("1/11/", "30/11/"), ("1/12/", "31/12/")]
    minas = months.index(month)
    #G = brandnew(dates[minas][0] + str(year), dates[minas][1] + str(year), category=category)

    G, finalnodes = slovenian(month, year, category=category,restriction=5)
    G=getbigcomp(G)
    melo=list(nx.isolates(G))
    G.remove_nodes_from(melo)
    nx.write_pajek(G, f"latenight.net")
    g = open(f"latenightsup.txt", 'w+')
    stri = ""
    # for node in finalnodes:
    #    stri=str(G.degree[node[0]]) +" "
    # print(G.degree[node[0]],file=g)
    temp=[]
    for node in finalnodes:
        if node[0] in list(G.nodes()):
            temp.append(node)
    finalnodes=temp
    for node in finalnodes:
        print(node[5], G.degree(node[0]), node[3], node[1], file=g)
    #print(stri,file=g)
    print("finished")
    g.close()


def allmonthspaj():
    months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
              "November", "December"]
    for year in range(2005,2016):
        for month in months:
            print(month,year)
            altern(month,year)
def help():
    months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
              "November", "December"]
    res=""
    for year in range(2005, 2016):
        for month in months:
            res+="\""+month+str(year)+"\""+","
    print(res)


def genesis(edgecoeff,nodecoef=0,country=0):
    edgescov=edgecoeff
    sizecov=nodecoef
    ethniccov=country
    #sizecov=nodecoef*(10**(nodepower))
    #print(sizecov)
    #sizecov=2.349*(math.exp(-6))
    #print(edgescov,sizecov)
    originalgraph,nodes=slovenian("August",2013)
    probs=[[None]*len(nodes)]*len(nodes)
    #print(len(probs),len(probs[1]))
    nodenames=[]
    sizes=[]
    for node in nodes:
        nodenames.append(node[0])
        sizes.append(node[5])
    #print(nodenames)
    #print(sizes)
    if country==0:
        for i in range(len(nodenames)):
            for j in range(i+1):
                logit=edgescov+sizecov*(sizes[i]+sizes[j])
                probs[i][j]=((math.e)**logit)/(1+(math.e)**(logit))
    else:
        for i in range(len(nodenames)):
            for j in range(i + 1):
                logit = edgescov + ethniccov * onefun(nodes[i][3] == nodes[j][3])+sizecov*(sizes[i]+sizes[j])
                probs[i][j] = ((math.e) ** logit) / (1 + (math.e) ** (logit))
    G=nx.Graph()
    for i in range(len(nodenames)):
        for j in range(i+1):
            p=np.random.uniform(0,1)
            if p<= probs[i][j]:
                G.add_edge(nodenames[i],nodenames[j])
    return (G,originalgraph)

def readfromstuff(filename):
    f=open(f"{filename}"+".net",'r')
    nodesize=(f.readline()).split(" ")[1]
    nodes=[]
    for i in range(int(nodesize)):
        temp=f.readline().split("\"")[1]
        nodes.append(temp)
    return nodes


def artcomp():
    artificial,_=genesis()
    months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
              "November", "December"]
    for year in range(2006,2016):
        for month in months:
            G,_=slovenian(month,year,category="regulated,financial")
            print(month,year,portrait_divergence(G,artificial))

def allnightbaby():
    f=open("nogovdir.txt",'r')
    g=open("guadalcanal.txt",'w+')

    c=f.readline()
    while(len(c)!=0):
        c=c.split()
        print(c)
        oyvey=c[1].split('20')
        oyvey=oyvey[0]
        print(oyvey)
        #oyvey=oyvey[:-1]=oyvey[-2:]
        print(oyvey[:-1],oyvey[-2:])
        l=f.readline()
        print(c,file=g)
        l=f.readline().split(' ')
        l = list(filter(('').__ne__, l))
        print(l)
        c=f.readline()
        artificial,_=genesis(float(l[0]),float(l[1]))
        months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October","November", "December"]
        for year in range(2006,2007):
            for month in months:
                newg,_=slovenian(month,year)
                print(month,year,portrait_divergence(artificial,newg),file=g)
    g.close()
    f.close()


def formstriangleundir(G,node):
    edges=list(G.edges())
    neighbors = [n for n in G.neighbors(node)]
    triangles=0
    for v in neighbors:
        neighbors.remove(v)
        for w in neighbors:
            if (v,w) in edges:
                triangles+=1
    return triangles


def quarters(quarter,year,category="None"):
    G=nx.DiGraph()
    thequarters=["Q1","Q2","Q3","Q4"]
    dates = [("1/1/", "31/3/"), ("1/4/", "30/6/"), ("1/7/", "30/9/"), ("1/10/", "31/12/")]
    quart = thequarters.index(quarter)
    thisshit=(dates[quart][0] + str(year), dates[quart][1] + str(year))
    nodes=newfaststats(thisshit[0], thisshit[1], category=category)
    #nodes=newfaststats('1/4/2006', '30/6/2006')
    trans = get_trans(dates[quart][0] + str(year), dates[quart][1] + str(year), restriction="None")
    nodesup = [(i[0].upper(), i[1], i[2], i[3], i[4],i[5]) for i in nodes]  # τα κάνω κεφαλαία
    transup = [(i[0].upper(), i[1].upper(), i[2], i[3]) for i in trans]  # τα κάνω κεφαλαία
    # print("stage3")
    transup = cleantrans(transup, nodesup)  # backup κατάλοιπο για κόμβους που δεν υπάρχουν στο nodes
    # trans=thecleanest(trans,nodes)
    transup = cleantransloop(transup)
    nodesup = cleannodes(nodesup, transup)  # στην περίεργη περίπτωση μεμονωμένου κόμβου που δε μας χρησιμεύει
    #print("nodes after cleaning",nodes)
    G = nodify(nodesup, G)  # φέρνω τους κόμβους σε καταλλληλη μορφή για το γράφημα
    G = edgify(transup, G)  # φέρνω τις ακμές σε κατάλληλη μορφή για το γράφημα
    category = {}
    for node in nodesup:
        category[node[0]] = node[1]
    nx.set_node_attributes(G, category, "category")
    # newplot(G)#κάλεσμα συνάρτησης από το plug.py για το σχεδιασμό
    # nx.draw(G, with_labels=True, node_size=1500, alpha=0.3, arrows=True)
    # plt.show()
    # dg.plot_2d()
    #print(nodesup,list(G.nodes()))
    return (G,nodesup)
    #G = controlroom(dates[minas][0] + str(year), dates[minas][1] + str(year), category=category)


def getquarter(q,year,category="None"):
    G, finalnodes = quarters(q, year, category=category)
    nx.write_pajek(G, f"quartersdir/{q}{year}.net")
    g = open(f"quarters*dir/{q}{year}sup.txt", 'w+')
    stri = ""
    # for node in finalnodes:
    #    stri=str(G.degree[node[0]]) +" "
    # print(G.degree[node[0]],file=g)
    # print(finalnodes)
    for node in finalnodes:
        print(node[5], G.degree(node[0]), node[3], file=g)
    print(stri, file=g)
    g.close()

def allquarts():
    allq=["Q1","Q2","Q3","Q4"]
    for year in range(2005,2016):
        for q in allq:
            print(q,year)
            getquarter(q,year,"None")

def trigenesis(edgecoef,trianglecoef):
    edgescov = edgecoef
    trianglecov = trianglecoef
    # sizecov=nodecoef*(10**(nodepower))
    # print(sizecov)
    # sizecov=2.349*(math.exp(-6))
    # print(edgescov,sizecov)
    originalgraph, nodes = slovenian("November", 2014)
    probs = [[None] * len(nodes)] * len(nodes)
    # print(len(probs),len(probs[1]))
    nodenames = []
    sizes = []
    for node in nodes:
        nodenames.append(node[0])
        sizes.append(node[5])
    # print(nodenames)
    # print(sizes)
    G=nx.Graph()
    G.add_nodes_from(nodenames)
    for i in range(len(nodenames)):
        for j in range(i + 1):
            logit = edgescov + trianglecov * nx.triangles(G,nodenames[i])
            probs[i][j] = ((math.e) ** logit) / (1 + (math.e) ** (logit))
            p = np.random.uniform(0, 1)
            if p <= probs[i][j]:
                G.add_edge(nodenames[i], nodenames[j])

    return (G, originalgraph)

def findalltriangles(G):
    sum=0
    for i in list(G.nodes()):
        sum+=nx.triangles(G,i)
        G.remove_node(i)
    return sum

def controlroom2(fromdate,todate,category="None",restriction="None",*args):
    if args==():
        G,thenodes=newplotting2(fromdate,todate,category,restriction)
    elif args[0]=="country":
        countryaggr(fromdate,todate,category)

    #G=newplotting(fromdate,todate,category)
    #newplot(G)
    #print(nx.adjacency_matrix(G))
    #print(type(nx.adjacency_matrix(G)))
    #nx.write_edgelist(loseweight(G),"edgelist.txt")
    return (G,thenodes)

def newplotting2(fromdate,todate,category="None",restriction="None",aggregation="None"):
    G = nx.DiGraph()
    #print("stage1")
    trans = get_trans(fromdate, todate, restriction)
    #print("trans",trans)
    nodes=newfaststatsx(fromdate, todate,category)
    #print("stage2")
    #nodes=cleannodes(nodes,trans)#αφαιρεί κόμβους που δεν έχουν ακμή(χρειάζομαι connected graph)
    nodesup=[(i[0].upper(),i[1],i[2],i[3],i[4],i[5]) for i in nodes]#τα κάνω κεφαλαία
    transup=[(i[0].upper(),i[1].upper(),i[2],i[3]) for i in trans]#τα κάνω κεφαλαία
    #print("stage3")
    transup = cleantrans(transup,nodesup) #backup κατάλοιπο για κόμβους που δεν υπάρχουν στο nodes
    #trans=thecleanest(trans,nodes)
    transup = cleantransloop(transup)
    nodesup = cleannodes(nodesup,transup) #στην περίεργη περίπτωση μεμονωμένου κόμβου που δε μας χρησιμεύει
    #print("these are trans homie:",len(transup),"these dem nodes",len(nodesup))
    #testingfun(nodes,trans)
    #print(nodesup)
    #trans = cleantrans(transup,nodesup) #backup κατάλοιπο για κόμβους που δεν υπάρχουν στο trans
    G=nodify(nodesup,G)#φέρνω τους κόμβους σε καταλλληλη μορφή για το γράφημα
    G=edgify(transup,G)#φέρνω τις ακμές σε κατάλληλη μορφή για το γράφημα
    #newplot(G)#κάλεσμα συνάρτησης από το plug.py για το σχεδιασμό
    #nx.draw(G, with_labels=True, node_size=1500, alpha=0.3, arrows=True)
    #plt.show()
    #dg.plot_2d()
    return (G,nodesup)

def karpf():
    G,finalnodes=controlroom2("1/1/2013","31/12/2015",category="regulated,financial")
    nx.write_pajek(G, "karpf.net")
    g = open("karpfsup.txt", 'w+')
    stri = ""
    # for node in finalnodes:
    #    stri=str(G.degree[node[0]]) +" "
    # print(G.degree[node[0]],file=g)
    for node in finalnodes:
        print(node[5], G.degree(node[0]), node[3], node[1], file=g)
    # print(stri,file=g)
    g.close()



def catgen(edgecoeff, nodecoef=0, country=0):
    cars = {'Regulated': [0,0.3958,4.3989],
            'Financial': [0.3958,0.9586,3.9513],
            'Governmental': []
            }
    df=pd.DataFrame()
    edgescov = edgecoeff
    sizecov = nodecoef
    ethniccov = country
    # sizecov=nodecoef*(10**(nodepower))
    # print(sizecov)
    # sizecov=2.349*(math.exp(-6))
    # print(edgescov,sizecov)
    originalgraph, nodes = slovenian("August", 2013)
    probs = [[None] * len(nodes)] * len(nodes)
    # print(len(probs),len(probs[1]))
    nodenames = []
    sizes = []
    for node in nodes:
        nodenames.append(node[0])
        sizes.append(node[5])
    # print(nodenames)
    # print(sizes)
    if country == 0:
        for i in range(len(nodenames)):
            for j in range(i + 1):
                logit = edgescov + sizecov * (sizes[i] + sizes[j])
                probs[i][j] = ((math.e) ** logit) / (1 + (math.e) ** (logit))
    else:
        for i in range(len(nodenames)):
            for j in range(i + 1):
                logit = edgescov + ethniccov * onefun(nodes[i][3] == nodes[j][3]) + sizecov * (sizes[i] + sizes[j])
                probs[i][j] = ((math.e) ** logit) / (1 + (math.e) ** (logit))
    G = nx.Graph()
    for i in range(len(nodenames)):
        for j in range(i + 1):
            p = np.random.uniform(0, 1)
            if p <= probs[i][j]:
                G.add_edge(nodenames[i], nodenames[j])
    return (G, originalgraph)

def newfaststatsx(fromdate, todate,
                     category="None"):  # return list of unique account holders [('node1,'category',
    if category == "None":
        addition = ""
    elif "," not in category:
        addition = f"\nand class.category = '{category}'"
    elif len(category.split(",")) == 2:
        things = category.split(",")
        addition = f"\nand (class.category = '{things[0]}' or class.category = '{things[1]}')"
    else:
        things = category.split(",")
        addition = f"\nand (class.category = '{things[0]}' or class.category = '{things[1]} or or class.category = '{things[2]}')"

    # print("Addition","l"+addition+"l")
    fromd = cleandate(fromdate)
    tod = cleandate(todate)
    sql = f"""select distinct tran.AcquiringAccountHolder, class.category, class.sector, acc.country, class.registry,size.avgTransNum
    from Transactions_New as tran, EUTL_AccountHolders as acc,EUTL_AccHolderClassification as class, EUTL_AccHolderTransactingSize as size
    where tran.AcquiringAccountHolder = acc.holderName
    and (tran.TransactionType='3-0' or tran.TransactionType='3-21' or tran.TransactionType='10-0')
    and acc.rawCode = size.holder
    and acc.rawCode = class.holder{addition}
    and tran.TransactionDate 
    between '{fromd[2]}-{fromd[1]}-{fromd[0]}' and '{tod[2]}-{tod[1]}-{tod[0]}'"""
    # print("stage1.1")
    sql2 = f"""select distinct tran.TransferringAccountHolder, class.category, class.sector, acc.country, class.registry,size.avgTransNum
    from Transactions_New as tran, EUTL_AccountHolders as acc,EUTL_AccHolderClassification as class, EUTL_AccHolderTransactingSize as size
    where tran.TransferringAccountHolder = acc.holderName
    and (tran.TransactionType='3-0' or tran.TransactionType='3-21' or tran.TransactionType='10-0')
    and acc.rawCode = size.holder
    and acc.rawCode = class.holder{addition}
    and tran.TransactionDate 
    between '{fromd[2]}-{fromd[1]}-{fromd[0]}' and '{tod[2]}-{tod[1]}-{tod[0]}'"""
    # print("stage1.2")
    # print(sql)
    cursor.execute(sql)
    acq = cursor.fetchall()
    # print("stage1.3")
    cursor.execute(sql2)
    tran = cursor.fetchall()
    # print("stage1.4")
    all = get_unique(acq + tran)
    # print("stage1.5")
    # print("nodes",all)
    return all

def get_trannies(fromdate,todate,restriction="None"):
    if restriction!="None":
        addition= f"and NbOfUnits>{restriction}"
    else: addition=""
    fromd=cleandate(fromdate)
    tod=cleandate(todate)
    key = 1 #1 for everything in, 2 for excluding NULL,
    if key ==1 :
        sql = f"""select TransferringAccountHolder, AcquiringAccountHolder, NbOfUnits, TransactionType, TransferringRegistry,AcquiringRegistry
        from Transactions_New where
        TransactionDate 
        between '{fromd[2]}-{fromd[1]}-{fromd[0]}' and '{tod[2]}-{tod[1]}-{tod[0]}'
        {addition}"""
    #If we want to exclude NULL appearances do the following
    elif key==2 :
        sql = f"""select TransferringAccountHolder, AcquiringAccountHolder, NbOfUnits, TransactionType, TransferringRegistry,AcquiringRegistry
        from Transactions_New
        where ((TransferringAccountHolder is not NULL) and (AcquiringAccountHolder is not NULL))
        and TransactionDate between '{fromd[2]}-{fromd[1]}-{fromd[0]}' and '{tod[2]}-{tod[1]}-{tod[0]}'
        {addition}"""
    cursor.execute(sql)
    tran=cursor.fetchall()
    final=[]
    if key==1:
        for i in tran:
            first=i[0]
            second=i[1]
            if i[0] is None:
                first=i[4]
            if i[1] is None:
                second=i[5]
            final.append((first,second,i[2],i[3],i[4]))
    return final


def brandnew(fromdate,todate,category="category",restriction="None"):
    G = nx.Graph()
    trans = get_trannies(fromdate, todate, restriction)
    #print("trans",trans)
    #nodes = get_unique_nodes(fromdate, todate, category)
    # nodes=cleannodes(nodes,trans)#αφαιρεί κόμβους που δεν έχουν ακμή(χρειάζομαι connected graph)
    #nodesup = [(i[0].upper(), i[1], i[2], i[3], i[4]) for i in nodes]  # τα κάνω κεφαλαία
    transup = [(i[0].upper(), i[1].upper(), i[2], i[3],i[4]) for i in trans]  # τα κάνω κεφαλαία
    #transup = cleantrans(transup, nodesup)  # backup κατάλοιπο για κόμβους που δεν υπάρχουν στο nodes
    # trans=thecleanest(trans,nodes)
    transup = cleantransloop(transup)
    #nodesup = cleannodes(nodesup, transup)  # στην περίεργη περίπτωση μεμονωμένου κόμβου που δε μας χρησιμεύει
    # testingfun(nodes,trans)
    # nx.draw(G, with_labels=True, node_size=1500, alpha=0.3, arrows=True)
    final=[]
    for i in transup:
        final.append((i[0],i[1]))
    G.add_edges_from(final)
    return G

def getbigcomp(G):
    comps = [c for c in sorted(nx.connected_components(G), key=len, reverse=True)]
    newtrans=cleanse(list(G.edges()),comps[0])
    newg=nx.Graph()
    newg.add_edges_from(newtrans)
    return newg

def getbigcompdir(G):
    comps = [c for c in sorted(nx.weakly_connected_components(G), key=len, reverse=True)]
    newtrans=cleanse(list(G.edges()),comps[0])
    newg=nx.Graph()
    newg.add_edges_from(newtrans)
    return newg

def cleanse(trans,nodes):
    tran=[]
    counter=0
    hashing=len(trans)*[1]
    nodenames=nodes
    for j in range(len(trans)):
            if trans[j][0] not in nodenames or trans[j][1] not in nodenames:
                hashing[j]=0
    for i in range(len(trans)):
           if hashing[i]==1:
                tran.append(trans[i])
    #print("counter",counter)
    #print(tran)
    return tran
def showallmonths():
    months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
              "November", "December"]
    dates = [("1/1/", "31/1/"), ("1/2/", "28/2/"), ("1/3/", "31/3/"), ("1/4/", "30/4/"), ("1/5/", "31/5/"),
             ("1/6/", "30/6/"), ("1/7/", "31/7/"), ("1/8/", "31/8/")
        , ("1/9/", "30/9/"), ("1/10/", "31/10/"), ("1/11/", "30/11/"), ("1/12/", "31/12/")]
    #G = controlroom(dates[minas][0] + str(year), dates[minas][1] + str(year))
    #edges=list(G.edges())
    #count=0
    #for edge in edges:
    #    count+=G.get_edge_data(edge[0],edge[1])["weight"]
    for year in range(2006,2017):
        for month in months:
            minas = months.index(month)
            G = brandnew(dates[minas][0] + str(year), dates[minas][1] + str(year))
            print(month,year)
            nt = Network(notebook=True)
            nt.from_nx(G)
            # nt.show_buttons(filter=['physics'])
            nt.show_buttons()
            nt.show(f'showtimecountryaggreg/{month}{year}.html')

def showallmonths2():
    months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
              "November", "December"]
    dates = [("1/1/", "31/1/"), ("1/2/", "28/2/"), ("1/3/", "31/3/"), ("1/4/", "30/4/"), ("1/5/", "31/5/"),
             ("1/6/", "30/6/"), ("1/7/", "31/7/"), ("1/8/", "31/8/")
        , ("1/9/", "30/9/"), ("1/10/", "31/10/"), ("1/11/", "30/11/"), ("1/12/", "31/12/")]
    #G = controlroom(dates[minas][0] + str(year), dates[minas][1] + str(year))
    #edges=list(G.edges())
    #count=0
    #for edge in edges:
    #    count+=G.get_edge_data(edge[0],edge[1])["weight"]
    for year in range(2006,2017):
        for month in months:
            minas = months.index(month)
            G,_ = slovenian(month,year)
            print(month,year)
            nt = Network(notebook=True)
            nt.from_nx(G)
            # nt.show_buttons(filter=['physics'])
            nt.show_buttons()
            nt.show(f'showtime/{month}{year}.html')

def showingfun(month,year,category="None"):
    print(month,year)
    months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
              "November", "December"]
    dates = [("1/1/", "31/1/"), ("1/2/", "28/2/"), ("1/3/", "31/3/"), ("1/4/", "30/4/"), ("1/5/", "31/5/"),
             ("1/6/", "30/6/"), ("1/7/", "31/7/"), ("1/8/", "31/8/")
        , ("1/9/", "30/9/"), ("1/10/", "31/10/"), ("1/11/", "30/11/"), ("1/12/", "31/12/")]
    minas = months.index(month)
    govnodes = newfaststats(dates[minas][0] + str(year), dates[minas][1] + str(year), category="governmental")
    finnodes = newfaststats(dates[minas][0] + str(year), dates[minas][1] + str(year), category="financial")
    regnodes = newfaststats(dates[minas][0] + str(year), dates[minas][1] + str(year), category="regulated")
    govnodes = [(i[0].upper(), i[1], i[2], i[3], i[4],i[5]) for i in govnodes]  # τα κάνω κεφαλαία
    finnodes = [(i[0].upper(), i[1], i[2], i[3], i[4],i[5]) for i in finnodes]  # τα κάνω κεφαλαία
    regnodes = [(i[0].upper(), i[1], i[2], i[3], i[4],i[5]) for i in regnodes]  # τα κάνω κεφαλαία
    regnodes = [node[0] for node in regnodes]
    finnodes = [node[0] for node in finnodes]
    govnodes = [node[0] for node in govnodes]

    newgovnodes=newfinnodes=newregnodes=[]
    G, finalnodes = slovenian(month, year, category)
    """for node in govnodes:
        if node in finalnodes:
            newgovnodes.append(node)
    for node in finnodes:
        if node in finalnodes:
            newfinnodes.append(node)
    for node in regnodes:
        if node in finalnodes:
            newregnodes.append(node)"""
    for node in list(G.nodes()):
        if node in regnodes:
            G.nodes[node]['group'] = 3
        elif node in finnodes:
            G.nodes[node]['group'] = 2
        elif node in govnodes:
            G.nodes[node]['group'] = 1
    nt = Network(notebook=True)
    nt.from_nx(G)
    # nt.show_buttons(filter=['physics'])
    nt.show_buttons()
    nt.show(f'showfundirected/{month}{year}.html')

def deggenesis(edgecoeff,nodecoef=0,country=0):
    edgescov=edgecoeff
    sizecov=nodecoef
    ethniccov=country
    #sizecov=nodecoef*(10**(nodepower))
    #print(sizecov)
    #sizecov=2.349*(math.exp(-6))
    #print(edgescov,sizecov)
    originalgraph,nodes=slovenian("August",2013,restriction=5)
    probs=[[None]*len(nodes)]*len(nodes)
    #print(len(probs),len(probs[1]))
    nodenames=[]
    sizes=[]
    for node in nodes:
        nodenames.append(node[0])
        sizes.append(node[5])
    #print(nodenames)
    #print(sizes)
    if country==0:
        for i in range(len(nodenames)):
            for j in range(i+1):
                logit=edgescov+sizecov*(originalgraph.degree(nodenames[i])+originalgraph.degree(nodenames[j]))
                probs[i][j]=((math.e)**logit)/(1+(math.e)**(logit))
    else:
        for i in range(len(nodenames)):
            for j in range(i + 1):
                logit = edgescov + ethniccov * onefun(nodes[i][3] == nodes[j][3])+sizecov*(originalgraph.degree(nodenames[i])+originalgraph.degree(nodenames[j]))
                probs[i][j] = ((math.e) ** logit) / (1 + (math.e) ** (logit))
    G=nx.Graph()
    for i in range(len(nodenames)):
        for j in range(i+1):
            p=np.random.uniform(0,1)
            if p<= probs[i][j]:
                G.add_edge(nodenames[i],nodenames[j])
    return (G,originalgraph)

def automation():
    direc=os.system("cd PycharmProjects/secondtry")
    print("`cd ~` ran with exit code %d" % direc)
    #for month in months:
    #    for year in range(2014,2016):
    #        toleda(month,year)
    for month in months:
        for year in range(2014,2015):
            gimme=os.system(f"./Directed_Graphlet_Counter_v3 {month}{year}.gw")

def monthwealth(month,year,category="None",restriction="None"):
    months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
              "November", "December"]
    dates = [("1/1/", "31/1/"), ("1/2/", "28/2/"), ("1/3/", "31/3/"), ("1/4/", "30/4/"), ("1/5/", "31/5/"),
             ("1/6/", "30/6/"), ("1/7/", "31/7/"), ("1/8/", "31/8/")
        , ("1/9/", "30/9/"), ("1/10/", "31/10/"), ("1/11/", "30/11/"), ("1/12/", "31/12/")]
    minas = months.index(month)
    #G = brandnew(dates[minas][0] + str(year), dates[minas][1] + str(year), category=category)

    G, finalnodes = slovenian(month, year, category=category,restriction=5)
    print("weighted",nx.is_weighted(G))
    #G=getbigcomp(G)
    #melo=list(nx.isolates(G))
    #G.remove_nodes_from(melo)
    strength=returnstrength(G)
    nx.write_pajek(G, f"sometesting/trying.net")
    g = open(f"sometesting/tryingsup.txt", 'w+')
    stri = ""
    # for node in finalnodes:
    #    stri=str(G.degree[node[0]]) +" "
    # print(G.degree[node[0]],file=g)
    temp=[]
    for node in finalnodes:
        if node[0] in list(G.nodes()):
            temp.append(node)
    finalnodes=temp
    print(len(strength),len(list(G.nodes())))
    for i in range(len(strength)):
        print(strength[i],file=g)
    #for node in finalnodes:
        #print(node[5], G.degree(node[0]), node[3], node[1], file=g)
    #print(stri,file=g)
    print("finished")
    g.close()

def strengthgenesis(edgecoeff,nodecoef=0,country=0):
    edgescov=edgecoeff
    sizecov=nodecoef
    ethniccov=country
    #sizecov=nodecoef*(10**(nodepower))
    #print(sizecov)
    #sizecov=2.349*(math.exp(-6))
    #print(edgescov,sizecov)
    originalgraph,nodes=slovenian("February",2014,restriction=5)
    print("originalgraphnodes",len(list(originalgraph.nodes())))
    strength=returnstrength(originalgraph)
    probs=[[None]*len(nodes)]*len(originalgraph.nodes())
    #print(len(probs),len(probs[1]))
    nodenames=[]
    sizes=[]
    for node in nodes:
        nodenames.append(node[0])
        sizes.append(node[5])
    #print(nodenames)
    #print(sizes)
    print("nodenames",len(nodenames))
    print("nodes", len(nodes))
    print("strength", len(strength))
    print("orignodes",len(originalgraph.nodes()))
    if country==0:
        for i in range(len(originalgraph.nodes())):
            for j in range(i+1):
                logit=edgescov+sizecov*(strength[i]+strength[j])
                probs[i][j]=((math.e)**logit)/(1+(math.e)**(logit))
    else:
        for i in range(len(originalgraph.nodes())):
            for j in range(i + 1):
                logit = edgescov + ethniccov * onefun(nodes[i][3] == nodes[j][3])+sizecov*(strength[i]+strength[j])
                probs[i][j] = ((math.e) ** logit) / (1 + (math.e) ** (logit))
    G=nx.Graph()
    for i in range(len(originalgraph.nodes())):
        for j in range(i+1):
            p=np.random.uniform(0,1)
            #print(p,probs[i][j])
            if p<= probs[i][j]:
                G.add_edge(nodenames[i],nodenames[j])
    return (G,originalgraph)

def returnstrength(G):
    strength = [0] * len(list(G.nodes()))
    counter = 0
    print("inside fun",len(list(G.nodes())))
    for node in list(G.nodes()):
        for neigh in G.neighbors(node):
            # strength[counter] += G.get_edge_data(node,neigh)
            strength[counter] += G[node][neigh]['weight']
        counter += 1
    print("inside fun",len(strength))
    print("counter",counter)
    return strength

def pieee():
    trans = get_trans("1/1/2012", "31/12/2015")
    myl = []
    for tran in trans:
        myl.append(tran[3])
    valuenames=get_unique(myl)
    numbers=[0]*len(valuenames)
    for elem in myl:
        numbers[valuenames.index(elem)]+=1
    print(valuenames)
    print(numbers)
    #print(df)
    #print(valuenames)
    #df = px.data.gapminder().query("year == 2007").query("continent == 'Europe'")
    #df.loc[df['pop'] < 2.e6, 'country'] = 'Other countries'  # Represent only large countries
    fig = px.pie(values=numbers,names=valuenames, title='Breakdown of types of companies')
    fig.show()

def savecommies(i):
    edges=[]
    G, _ = slovenian("January", 2015)
    for edge in list(G.edges()):
        if G[edge[0]][edge[1]]["weight"] <= 15:
            edges.append(edge)
    print(len(list(G.edges())), len(edges))
    G.remove_edges_from(edges)
    G1 = getbigcomp(G)
    c = asyn_fluidc(G1,2)
    #print(c)
    commies=[com for com in c]
    lens = [len(com) for com in commies]
    print(lens)
    nodelist1,nodelist2=commies[0],commies[1]
    #print(nodelist1,nodelist2)
    newG=G.subgraph(nodelist1)
    newG1=G.subgraph(nodelist2)
    nx.write_edgelist(newG,f"sub1edgelist{i}")
    nx.write_edgelist(newG1, f"sub1edgelist{i+1}")
    nx.write_pajek(newG, f"sub1edgelist{i}.net")
    nx.write_pajek(newG1, f"sub1edgelist{i+1}.net")
    f1=open(f"sub1edgelist{i}sup.txt","w+")
    cat1=list(nx.get_node_attributes(newG,"category").values())
    cat2=list(nx.get_node_attributes(newG1,"category").values())
    f2 = open(f"sub1edgelist{i+1}sup.txt", "w+")
    for i in cat1:
        print(i,file=f1)
    for i in cat2:
        print(i,file=f2)
    """for node in list(newG.nodes()):
        if newG.nodes[node]["category"]=="regulated":
            newG.nodes[node]['group'] = 1
        elif newG.nodes[node]["category"]=="financial":
            newG.nodes[node]['group'] = 2
        elif newG.nodes[node]["category"]=="governmental":
            newG.nodes[node]['group'] = 3
    nt = Network(notebook=True)
    nt.from_nx(newG)
    # nt.show_buttons(filter=['physics'])
    nt.show_buttons()
    nt.show(f'subgraph1.html')
    for node in list(newG1.nodes()):
        if newG1.nodes[node]["category"]=="regulated":
            newG1.nodes[node]['group'] = 1
        elif newG1.nodes[node]["category"]=="financial":
            newG1.nodes[node]['group'] = 2
        elif newG1.nodes[node]["category"]=="governmental":
            newG1.nodes[node]['group'] = 3
    nt = Network(notebook=True)
    nt.from_nx(newG1)
    # nt.show_buttons(filter=['physics'])
    nt.show_buttons()
    nt.show(f'subgraph2.html')"""

def metakarpf():
    thequarters=["Q1","Q2","Q3","Q4"]
    for year in range(2013,2016):
        for quart in thequarters:
            print(quart, year)
            newG,finalnodes=quarters(quart,year)
            for node in list(newG.nodes()):
                if newG.nodes[node]["category"] == "regulated":
                    newG.nodes[node]['group'] = 1
                elif newG.nodes[node]["category"] == "financial":
                    newG.nodes[node]['group'] = 2
                elif newG.nodes[node]["category"] == "governmental":
                    newG.nodes[node]['group'] = 3
            nt = Network(notebook=True)
            nt.from_nx(newG)
            # nt.show_buttons(filter=['physics'])
            nt.show_buttons()
            nt.show(f'metakarpfquart/{quart}{year}.html')

def metakarpfy():
    for year in range(2013,2016):
        print(year)
        newG=controlroom(f"1/1/{year}",f"31/12/{year}")
        for node in list(newG.nodes()):
            if newG.nodes[node]["category"] == "regulated":
                newG.nodes[node]['group'] = 1
            elif newG.nodes[node]["category"] == "financial":
                newG.nodes[node]['group'] = 2
            elif newG.nodes[node]["category"] == "governmental":
                newG.nodes[node]['group'] = 3
        nt = Network(notebook=True)
        nt.from_nx(newG)
        # nt.show_buttons(filter=['physics'])
        nt.show_buttons()
        nt.show(f'metakarpfquart/{year}.html')

def bi():
    G,_=slovenian("January",2014)
    count=0
    for edge in list(G.edges()):
        if ((edge[1],edge[0]) in list(G.edges())):
            count+=1
    print(count)

def alldupl():
    dates = [("1/1/", "31/1/"), ("1/2/", "28/2/"), ("1/3/", "31/3/"), ("1/4/", "30/4/"), ("1/5/", "31/5/"),
             ("1/6/", "30/6/"), ("1/7/", "31/7/"), ("1/8/", "31/8/")
        , ("1/9/", "30/9/"), ("1/10/", "31/10/"), ("1/11/", "30/11/"), ("1/12/", "31/12/")]
    f=open("alldupl.txt","w+")
    for month in months:
        sent=month
        for year in range(2006,2016):
            minas = months.index(month)
            print(year,file=f)
            trans=get_trans(dates[minas][0] + str(year), dates[minas][1] + str(year))
            count=0
            for tran in trans:
                if (tran[1],tran[0],tran[2],tran[3]) in trans:
                    count+=1
            sent+=" " +str(round(count/len(trans) *100,2))+"\% & "
        print(sent,file=f)

def mixingartist(edgecoeff,nodecoeff=0,mix=[0,0,0,0,0,0],country=0):
    edgescov = edgecoeff
    sizecov = nodecoeff
    ethniccov = country
    # sizecov=nodecoef*(10**(nodepower))
    # print(sizecov)
    # sizecov=2.349*(math.exp(-6))
    # print(edgescov,sizecov)
    originalgraph, nodes = slovenian("February", 2014, restriction=5)
    print("originalgraphnodes", len(list(originalgraph.nodes())))
    strength = returnstrength(originalgraph)
    probs = [[None] * len(nodes)] * len(originalgraph.nodes())
    # print(len(probs),len(probs[1]))
    print(nodes)
    nodenames = []
    sizes = []
    for node in nodes:
        nodenames.append(node[0])
        sizes.append(node[5])
    # print(nodenames)
    # print(sizes)
    print("nodenames", len(nodenames))
    print("nodes", len(nodes))
    print("strength", len(strength))
    print("orignodes", len(originalgraph.nodes()))
    if country == 0:
        for i in range(len(originalgraph.nodes())):
            for j in range(i + 1):

                mixing=mix[0]*onefun((nodes[i][1]=="financial")and(nodes[j][1]=="financial"))+mix[1]*onefun((nodes[i][1]=="financial")and(nodes[j][1]=="governmental"))
                +mix[2]*onefun((nodes[i][1]=="governmental")and(nodes[j][1]=="governmental"))+mix[3]*onefun((nodes[i][1]=="financial")and(nodes[j][1]=="regulated"))
                +mix[4]*onefun((nodes[i][1]=="governmental")and(nodes[j][1]=="regulated"))+mix[5]*onefun((nodes[i][1]=="regulated")and(nodes[j][1]=="regulated"))
                logit = edgescov + sizecov * (strength[i] + strength[j]) + mixing
                probs[i][j] = ((math.e) ** logit) / (1 + (math.e) ** (logit))
    else:
        for i in range(len(originalgraph.nodes())):
            for j in range(i + 1):
                mixing = mix[0] * onefun(
                    (nodes[i][1] == "financial") and (nodes[j][1] == "financial")) + mix[1] * onefun(
                    (nodes[i][1] == "financial") and (nodes[j][1] == "governmental"))
                +mix[2] * onefun(
                    (nodes[i][1] == "governmental") and (nodes[j][1] == "governmental")) + mix[
                    3] * onefun((nodes[i][1] == "financial") and (nodes[j][1] == "regulated"))
                +mix[4] * onefun((nodes[i][1] == "governmental") and (nodes[j][1] == "regulated")) + \
                mix[5] * onefun((nodes[i][1] == "regulated") and (nodes[j][1] == "regulated"))

                logit = edgescov + ethniccov * onefun(nodes[i][3] == nodes[j][3]) + sizecov * (
                            strength[i] + strength[j])+mixing
                probs[i][j] = ((math.e) ** logit) / (1 + (math.e) ** (logit))
    G = nx.Graph()
    for i in range(len(originalgraph.nodes())):
        for j in range(i + 1):
            p = np.random.uniform(0, 1)
            # print(p,probs[i][j])
            if p <= probs[i][j]:
                G.add_edge(nodenames[i], nodenames[j])
    return (G, originalgraph)

start=time.time()
months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
              "November", "December"]
#get_trans("29/6/2014","30/6/2014")
#get_unique_nodes("29/6/2014","30/6/2014")
#plotting("29/6/2014","30/7/2014")
#plotting("29/3/2014","3/5/2014")
#a = np.random.randint(0, 5, size=(15, 15))
#D = nx.DiGraph(a)
#nx.draw(D, with_labels=True, node_size=1500, alpha=0.3, arrows=True)
#countryaggr("29/6/2014","1/7/2014")
#f=open("temporary.txt","w+")
#massstats("1/1/2014", "1/2/2014","regulated,financial")
#f.close()
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
#pyvistest()
#visual("September",2012,"regulated,financial")
#weightdist("1/3/2012","31/3/2012","governmental")
#controlroom("1/9/2012","30/9/2012","governmental,regulated")
#doingrand()
#toleda("November",2011)
#sotiris()
#manytests2()
#tempgraph()
#doit()
#reggovfin("January",2011)
#reggovfin("February",2011)
#reggovfin("March",2011)
#trials("June",2011)
#vol("June",2011)
#nightly()
#runtrials()
#whatamidoing()
#centricbetter()
#damnindians("May","2013")
"""list1=newfaststats("1/1/2005","31/12/2015")
print(len(list1))
list2=newfaststats("1/1/2005","31/12/2015","regulated")
print(len(list2))
list3=newfaststats("1/1/2005","31/12/2015","governmental")
print(len(list3))
list4=newfaststats("1/1/2005","31/12/2015","financial")
print(len(list4))"""
#damnindians("August",2012)
#G=controlroom("1/12/2011","31/12/2011")
#nx.write_pajek(G,"madness.net")
#print(help(pajek_tools.PajekWriter))
#damnindians("January",2011)
#altern("July",2012)
#allmonthspaj()
#help()
#artificial,originalgraph= genesis()
#G,_=slovenian("August",2015)
#central = nx.betweenness_centrality(G)
#central2=dict(sorted(central.items(),key= lambda x:x[1]))
#print(central2)
#G,originalgraph= genesis()
#originalgraph,_=slovenian("August",2012)
#print(portrait_divergence(G,artificial))
#readfromstuff("directedsizenumber/August2013")
#altern("November",2015)
#artcomp()
#allnightbaby()
#altern("June",2009)
def garbage():
    artificial,_=trigenesis(-5.95972,2.74810)
    melo=list(nx.isolates(artificial))
    print(melo)
    artificial.remove_nodes_from(melo)
    nt = Network(notebook=True)
    nt.from_nx(artificial)
    #nt.show_buttons(filter=['physics'])
    nt.show_buttons()
    nt.show('arti.html')
    G1=artificial
    #print(findalltriangles(artificial))
    G,_=slovenian("November",2014)
    G2=G
    #print(findalltriangles(G))
    print("connected", nx.is_connected(artificial))
    print("den1",nx.density(artificial),"den2",nx.density(G))
    print(portrait_divergence(G1,G2))
#print(list(G.nodes())[4])
#print(type(nx.triangles(G,"CELANESE PRODUCTION GERMANY GMBH & CO. KG")))
#allmonthspaj()
#karpf()
#print(get_trannies("1/6/2011","10/6/2011"))
#art,_=genesis(-7.195,0,country=0.374)
#G,_=slovenian("August",2012)
def whocares():
    G=brandnew("1/6/2011","30/6/2011")
    G=getbigcomp(G)
    nt = Network(notebook=True)
    nt.from_nx(G)
    #nt.show_buttons(filter=['physics'])
    nt.show_buttons()
    nt.show('arti.html')
    g = ig.Graph.from_networkx(G)
    """fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
    ax1.set_title("Plot with igraph plot")
    layout = g.layout_kamada_kawai()
    ig.plot(g, layout=layout, target=ax1)
    plt.axis("off")
    plt.show()"""
    comps=[c for c in sorted(nx.connected_components(G), key=len, reverse=True)]
    print(comps[0])
    print([len(c) for c in sorted(nx.connected_components(G), key=len, reverse=True)])

def plot_degree_dist(G):
    edges = []
    degrees = [G.degree(n) for n in G.nodes()]
    logd=[math.log(degree) for degree in degrees]
    plt.hist(degrees,range=(0,max(logd)))
    plt.show()

#cust=custompaj("March",2014)
#showingfun("March", 2014)
"""art,_=genesis(-7.104,-4.050e-07)
orig,_=slovenian("March",2015)
newplot(orig)
orig=getbigcomp(orig)
print(portrait_divergence(art,orig))"""
#showallmonths()
#showallmonths2()
#art,_=trigenesis(-7.28601 ,4.02247)
#art,_=genesis(-7.122,2.221e-06)
#art,_=deggenesis(-7.1455038,-0.0009901)
#showingfun("February",2013)
#G=brandnew("1/3/2014","31/3/2014")
#G=getbigcomp(G)
#G1=brandnew("1/3/2015","31/3/2015")
#G1=getbigcomp(G1)
#print(portrait_divergence(G,G1))
#print([len(c) for c in sorted(nx.connected_components(G), key=len, reverse=True)])
#for year in range(2012,2015):
#    for month in months:
#    showingfun("August",year)
#automation()
#monthwealth("February",2014)
#G,_=slovenian("February",2014)
#print("orig",nx.is_weighted(G))
#art,_=strengthgenesis(-8.904,4.357e-08)
#print("art",nx.is_weighted(art))
#showingfun("May",2014)

"""G,_=slovenian("August",2014)
G1=nx.Graph()
#ass=['big']*len(list(G.nodes()))
#nx.set_node_attributes(G, ass, "ass")
ass={}
for node in list(G.nodes()):
    #G1.add_node(node,ass='big')
    #print(G[node]["ass"])
    ass[node]="big"
nx.set_node_attributes(G, ass, "ass")
nodes=list(G.nodes())
print(G.nodes[random.choice(nodes)]["ass"])"""
def latestfunction():
    for month in months:
        print(month)
        G,_=slovenian(month,2015)
        data = nx.attribute_mixing_matrix(G, "category", mapping={"regulated": 0, "governmental": 1, "financial": 2})

        #data = np.asarray(data)
    #print(data)

        fig = plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
        plt.clf()
        ax = fig.add_subplot(111)
        ax.set_aspect(1)
        res = sns.heatmap(data, annot=True, fmt='.3f', cmap="YlGnBu", vmin=0.0, vmax=1.0)
        plt.title(f'Mixing Matrix for Categories for {month} 2015',fontsize=12)
        plt.xticks([i+0.5 for i in range(data.shape[0])], ['reg', 'gov', 'fin'])
        plt.xticks(rotation=0)

        plt.yticks([i+0.5 for i in range(data.shape[1])], ['reg', 'gov', 'fin'])
        plt.yticks(rotation=0)
        plt.savefig(f"heatmap{month}2015.png", bbox_inches='tight', dpi=100)
        #plt.show()

#result = pyreadr.read_r('varsALLTrans_netStats.RData')
#print(type(result))
#print(result.keys())
#print(result["vertexStats_annually"])
"""G,_=slovenian("January",2015)
c=list(greedy_modularity_communities(getbigcomp(G)))
#c = asyn_fluidc(G,2)

lens=[len(com) for com in c]
print(lens)
print(nx.is_weighted(G))

print(lens)"""
#print(nx.center(getbigcomp(G)))
#G=getbigcomp(G)
#print(nx.density(G),len(list(G.nodes())))
#G1,_=slovenian("April",2015)
#plot_degree_dist(G1)
#karpf()

"""G,_=slovenian("May",2014)
print(nx.density(G))
#G=getbigcomp(G)
G1,_=slovenian("May",2015)
print(nx.density(G1))
#G1=getbigcomp(G1)
print(nx.density(G),nx.density(G1))
print(portrait_divergence(G,G1))
print(portrait_divergence())"""
#print(nx.is_weighted(G),nx.is_weighted(G1))
#print(portrait_divergence(G,G1))
#print(portrait_divergence(art,G))
#print(portrait_divergence(art,loseweight(G)))
#pieee()
#for tran in trans:
#savecommies()

"""G,_=slovenian("January",2015)
edges=[]
for edge in list(G.edges()):
    if G[edge[0]][edge[1]]["weight"] <= 15:
        edges.append(edge)
print(len(list(G.edges())), len(edges))
G.remove_edges_from(edges)
G1 = getbigcomp(G)
c = kernighan_lin_bisection(G1)
lengths=[len(i) for i in c]
print(lengths)"""
#G,_=slovenian("January",2015)
#nx.write_edgelist(G,"test_edgelist")
#print(nx.clustering(G))
#dic2=dict(sorted(nx.clustering(G).items(),key= lambda x:x[1]))
#print(dic2)
#print(sorted(nx.clustering(G)))
#metakarpfy()
#alldupl()
#coms = algorithms.louvain(G, weight='weight', resolution=1., randomize=False)
#coms = algorithms.leiden(G)

#comms=[com for com in coms]
#print(coms)

#viz.plot_community_graph(G, coms)
#plt.show()
"""for i in range(0,10,2):
    print(i)
    savecommies(i)"""

#print(G[nodes[3]]["category"])
def pars(file):
    f=open(file)
    verts= f.readline().split(" ")[1]
    print(verts)
    categ=[]
    for vert in range(int(verts)):
        line=f.readline().split("\"")
        lastpart=line[2].split(" ")
        cat=lastpart[-1]
        categ.append(cat)
    return categ

def myread(filename):
    f=open(filename,"r")
    verts = f.readline().split(" ")[1]
    print(verts)
    categ = []
    nodenames= []
    codes= []
    for vert in range(int(verts)):
        line=f.readline().split("\"")
        nodenames.append(line[1])
        codes.append(line[0])
        print(line[0])
        lastpart=line[2].split(" ")
        cat=lastpart[-1]
        categ.append(cat)
    f.readline()
    edges=[]
    while(True):
        lineread=f.readline()
        print(lineread)
        if lineread=="":
            break
        line=lineread.split(" ")
        print(line)
        edges.append((nodenames[codes.index(line[0])],nodenames[codes.index(line[1])],line[2]))
    G.add_weighted_edges_from(edges)
    nx.set_node_attributes(G, categ, "category")
    return G

"""categ=pars("sub1edgelist0.net")
fh = open("sub1edgelist", "r")
#G=nx.read_pajek("sub1edgelist0.net")
G = nx.read_edgelist(fh)
fh.close()
nodes=list(G.nodes())
nx.set_node_attributes(G, categ, "category")"""
#arti,_=mixingartist(-6.9531,mix=[2.9945,4.7771,0,2.2905,2.5525,0])
#print(onefun((1==1) and (2==2)))
G=myread("sub1edgelist0.net")