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



def plotting(datefrom,dateto):
    sql=f"""select tran.acquiringaccountholder
        from transactions_new as tran, eutl_accountholders as acc, eutl_accholderclassification as class
        where tran.acquiringaccountholder = """
f"""select distinct tran.acquiringaccountholder
        from transactions_new as tran, eutl_accountholders as acc, eutl_accholderclassification as class
        where tran.acquiringaccountholder = acc.holdername
        and acc.rawcode=class.holder
        and category='financial'"""

def cleandate(date):
    clean=date.split("/")
    return clean

def get_unique_nodes(fromdate,todate):#μοναδικοί
    fromd = cleandate(fromdate)
    tod = cleandate(todate)
    sql = f"""select tran.acquiringaccountholder, class.category, class.sector, acc.country, class.registry
    from transactions_new as tran, eutl_accountholders as acc,eutl_accholderclassification as class
    where tran.acquiringaccountholder = acc.holdername
    and acc.rawcode = class.holder
    and tran.transactiondate 
    between '{fromd[2]}-{fromd[1]}-{fromd[0]}' and '{tod[2]}-{tod[1]}-{tod[0]}'"""
    sql2 = f"""select tran.transferringaccountholder, class.category, class.sector
    from transactions_new as tran, eutl_accountholders as acc,eutl_accholderclassification as class
    where tran.transferringaccountholder = acc.holdername
    and acc.rawcode = class.holder
    and tran.transactiondate 
    between '{fromd[2]}-{fromd[1]}-{fromd[0]}' and '{tod[2]}-{tod[1]}-{tod[0]}'"""
    cursor.execute(sql)
    acq = cursor.fetchall()
    cursor.execute(sql2)
    tran = cursor.fetchall()
    all=get_unique(acq+tran)

    #print(tran)
    #print(acq)
    #print(all)
    return all

def getnodesalt(trans):
    nodes=[]
    for i in trans:
        nodes.append(i[0])
        nodes.append(i[1])
    cleannodes= get_unique(nodes)
    return cleannodes

def get_trans(fromdate,todate): #συναλλαγές μεταξύ δυο ημερομηνιών
    fromd=cleandate(fromdate)
    tod=cleandate(todate)
    sql = f"""select transferringaccountholder, acquiringaccountholder, nbofunits, transactiontype
    from transactions_new where transactiondate 
    between '{fromd[2]}-{fromd[1]}-{fromd[0]}' and '{tod[2]}-{tod[1]}-{tod[0]}'"""
    #sql = f"""select transactiondate from transactions_new limit 10"""
    cursor.execute(sql)
    tran=cursor.fetchall()
    #print(tran)
    #print(type(tran[1]),tran[1])
    return tran
def test():
    sql00 = f"""select distinct tran.acquiringaccountholder
            from transactions_new as tran, eutl_accountholders as acc, eutl_accholderclassification as class
            where tran.acquiringaccountholder = acc.holdername
            and acc.rawcode=class.holder"""
    sql01 = f"""select distinct tran.acquiringaccountholder
            from transactions_new as tran, eutl_accountholders as acc
            where tran.acquiringaccountholder = acc.holdername"""
    cursor.execute(sql00)
    test=cursor.fetchall()
    cursor.execute(sql01)
    test2 = cursor.fetchall()
    print(len(test),len(test2))

def graphify(dic):
    df = pd.DataFrame(dic)
    G = nx.from_pandas_edgelist(df, 'from', 'to', create_using=nx.DiGraph())
    nx.draw(G, with_labels=False, node_size=1500, alpha=0.3, arrows=True)

def graphing(fromdate,todate):
    dic={'from':[],'to':[]}
    trans=get_trans(fromdate,todate)
    dicoutnodes=dicify(trans)
    dicinnodes=revdicify(trans)
    nodes=getnodesalt(trans)
    #nodes=get_unique_nodes(fromdate,todate)
    #print(nodes)
    for i in range(0,len(trans)):
        dic['from'].append(trans[i][0])
        dic['to'].append(trans[i][1])
    df=pd.DataFrame(dic)
    #G = nx.from_pandas_edgelist(df, 'from', 'to', create_using=nx.DiGraph())
    #nx.draw(G, with_labels=True, node_size=1500, alpha=0.3, arrows=True)
    #plt.show()
    #degreedistr(dicoutnodes)
    #degreedistr(dicinnodes)
    mat=getmatrix(trans,nodes)
    simplematrix = getsimplematrix(mat)
    G = nx.DiGraph(simplematrix)
    density = nx.density(G)
    #print(density)
    #nx.draw(G, with_labels=True, node_size=1500, alpha=0.3, arrows=True)
    nx.draw_kamada_kawai(G,alpha=0.3)
    plt.show()
    centr=nx.betweenness_centrality(G, k=None, normalized=True, weight=None, endpoints=False, seed=None)
    pagerank=nx.pagerank(G,alpha=0.8)
    #communities = community.greedy_modularity_communities(G)
    #print(communities)

#def adjac(dic):
    #for i in dic.keys():
def getmatrix(trans,nodes):
    empty=len(nodes)*[0]
    #mat=len(nodes)*[len(nodes)*[0]]
    mat = np.random.randint(0, 1, size=(len(nodes), len(nodes)))

    cleannodes=nodes
    #print(cleannodes)
    #for i in range(len(nodes)):
    #    cleannodes.append(nodes[i][0])
    for i in range(0,len(nodes)):
        for j in range(0,len(trans)):
            if nodes[i] == trans[j][0]:
                mat[i][cleannodes.index(trans[j][1])] = (trans[j][2])
    #print(mat)

    return mat

    #print(mat)

def getsimplematrix(matrix):
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
    #print((len(acq),len(tran)))
    for i in range(0, len(tran)):
        dic['from'].append(tran[i][key])
        dic['to'].append(acq[i][key])
    df = pd.DataFrame(dic)
    G = nx.from_pandas_edgelist(df, 'from', 'to', create_using=nx.DiGraph())
    nx.draw(G, with_labels=True, node_size=1500, alpha=0.3, arrows=True)
    plt.show()



ignite()
#get_trans("29/6/2014","30/6/2014")
#test()
#get_unique_nodes("29/6/2014","30/6/2014")
#plt.show()
graphing("29/6/2014","30/7/2014")
#nodeaggr("29/6/2014","30/7/2014","countrypollution")
#graphing("29/6/2014","20/7/2015")
#a = np.random.randint(0, 5, size=(15, 15))
#D = nx.DiGraph(a)
#nx.draw(D, with_labels=True, node_size=1500, alpha=0.3, arrows=True)
#plt.show()
