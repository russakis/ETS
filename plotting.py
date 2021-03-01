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
from random import random

def controlroom(fromdate,todate,*args):
    print(args)
    if args==():
        newplotting(fromdate,todate)
    elif args[0]=="country":
        countryaggr(fromdate,todate)

#η γενική συνάρτηση στην οποία συμβαίνουν τα πάντα
def newplotting(fromdate,todate):
    G = nx.DiGraph()
    trans = get_trans(fromdate, todate)
    #print("trans",trans)
    nodes=get_unique_nodes(fromdate, todate)
    nodesup=[(i[0].upper(),i[1],i[2],i[3],i[4]) for i in nodes]#τα κάνω κεφαλαία
    transup=[(i[0].upper(),i[1].upper(),i[2],i[3]) for i in trans]#τα κάνω κεφαλαία
    #testingfun(nodes,trans)
    trans = cleantrans(transup,nodesup) #backup κατάλοιπο για κόμβους που δεν υπάρχουν στο trans
    G=nodify(nodesup,G)#φέρνω τους κόμβους σε καταλλληλη μορφή για το γράφημα
    G=edgify(trans,G)#φέρνω τις ακμές σε κατάλληλη μορφή για το γράφημα
    newplot(G)#κάλεσμα συνάρτησης από το plug.py για το σχεδιασμό
    #nx.draw(G, with_labels=True, node_size=1500, alpha=0.3, arrows=True)
    #plt.show()

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

def get_unique_nodes(fromdate,todate):#return list of unique account holders [('node1,'category',
    fromd = cleandate(fromdate)
    tod = cleandate(todate)
    sql = f"""select distinct tran.acquiringaccountholder, class.category, class.sector, acc.country, class.registry
    from transactions_new as tran, eutl_accountholders as acc,eutl_accholderclassification as class
    where tran.acquiringaccountholder = acc.holdername
    and acc.rawcode = class.holder
    and tran.transactiondate 
    between '{fromd[2]}-{fromd[1]}-{fromd[0]}' and '{tod[2]}-{tod[1]}-{tod[0]}'"""
    sql2 = f"""select distinct tran.transferringaccountholder, class.category, class.sector, acc.country, class.registry
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
    #print("nodes",all)
    return all

def getnodesalt(trans): #get nodes from transactions as list ["node1","node2"...]
    nodes=[]
    for i in trans:
        nodes.append(i[0])
        nodes.append(i[1])
    cleannodes= get_unique(nodes)
    return cleannodes

def get_trans(fromdate,todate): #συναλλαγές μεταξύ δυο ημερομηνιών
    fromd=cleandate(fromdate)
    tod=cleandate(todate)
    key = 2 #1 for everything in, 2 for excluding NULL,
    if key ==1 :
        sql = f"""select transferringaccountholder, acquiringaccountholder, nbofunits, transactiontype
        from transactions_new where transactiondate 
        between '{fromd[2]}-{fromd[1]}-{fromd[0]}' and '{tod[2]}-{tod[1]}-{tod[0]}'"""
    #If we want to exclude NULL appearances do the following
    elif key==2 :
        sql = f"""select transferringaccountholder, acquiringaccountholder, nbofunits, transactiontype
        from transactions_new
        where ((transferringaccountholder is not NULL) and (acquiringaccountholder is not NULL))
        and transactiondate between '{fromd[2]}-{fromd[1]}-{fromd[0]}' and '{tod[2]}-{tod[1]}-{tod[0]}'"""
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


def cleantrans(trans,nodes):
    tran=[]
    hashing=len(trans)*[0]
    for j in range(len(trans)):
        for i in range(len(nodes)):
            if nodes[i][0] == trans[j][0] or nodes[i][0] == trans[j][1]:
                hashing[j]=1
    for i in range(len(trans)):
        if hashing[i]==1:
            tran.append(trans[i])
    #print(tran)
    return tran

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
#controlroom("29/6/2014","1/7/2014","country")
