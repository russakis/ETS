import mysql.connector
from mysqlconfig import cursor, db
import datetime
import numpy as np
import matplotlib.pyplot as plt



#def add_log(text,user):
 #   sql=("INSERT ")

def ignite():#διαλέγω τη βάση
    sql = ("USE EU_ETS")
    cursor.execute(sql)

def get_unique(listy):#μοναδικά στοιχεία σε λίστα

    list_of_unique = []

    unique= set(listy)

    for el in unique:
        list_of_unique.append(el)

    return list_of_unique


def checkdif(table1,table2):#έλεγχος μεγεθών πινάκων
    sql1 = f"select * from {table1}"
    cursor.execute(sql1)
    size1=len(cursor.fetchall())
    sql2 = f"select * from {table2}"
    cursor.execute(sql2)
    size2 = len(cursor.fetchall())
    print("size1",size1)
    print("size2",size2)

def counting(table1):#μέγεθος πίνακα
    sql1 = f"select * from {table1}"
    cursor.execute(sql1)
    size1 = len(cursor.fetchall())
    print("size:",size1)

def mineral():#διυλιστήρια
    refer = ('2014-01-', '2014-02-', '2014-03-', '2014-04-', '2014-05-', '2014-06-', '2014-07-', '2014-08-', '2014-09-',
             '2014-10-', '2014-11-', '2014-12-')
    mineralx = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    mineraly = []
    for month in refer:
        sql00 = f"""select distinct tran.acquiringaccountholder
        from transactions_new as tran, eutl_accountholders as acc, eutl_accholderclassification as class
        where tran.acquiringaccountholder = acc.holdername
        and acc.rawcode=class.holder
        and TransactionDate LIKE '{month}%'"""
        sql01 = f"""select distinct tran.transferringaccountholder
        from transactions_new as tran, eutl_accountholders as acc, eutl_accholderclassification as class
        where tran.transferringaccountholder = acc.holdername
        and acc.rawcode=class.holder
        and TransactionDate LIKE '{month}%'"""
        cursor.execute(sql00)
        acqall = cursor.fetchall()
        cursor.execute(sql01)
        transall = cursor.fetchall()
        duplall = acqall + transall
        sql = f"""select tran.acquiringaccountholder
                from transactions_new as tran, eutl_accountholders as acc, eutl_accholderclassification as class
                where tran.acquiringaccountholder = acc.holdername
                and acc.rawcode=class.holder
                and (class.sector='2' or class.sector='21')
                and TransactionDate LIKE '{month}%'"""
        sql1 = f"""select tran.transferringaccountholder
                from transactions_new as tran, eutl_accountholders as acc, eutl_accholderclassification as class
                where tran.transferringaccountholder = acc.holdername
                and acc.rawcode=class.holder
                and (class.sector='2' or class.sector='21')
                and TransactionDate LIKE '{month}%'"""
        cursor.execute(sql)
        acq = cursor.fetchall()
        cursor.execute(sql1)
        trans = cursor.fetchall()
        dupllist= acq+trans
        mineraly.append(len(get_unique(dupllist))*100/len(get_unique(duplall)))
        #print(len(acq),len(trans),len(get_unique(dupllist)))
    sec[0].plot(mineralx, mineraly)

def sector20():#σεκτορ 20 για ολλανδία γαλλία γερμανία
    refer = ('2014-01-', '2014-02-', '2014-03-', '2014-04-', '2014-05-', '2014-06-', '2014-07-', '2014-08-', '2014-09-',
             '2014-10-', '2014-11-', '2014-12-')
    combx = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    comby = []
    for month in refer:
        sql00 = f"""select tran.acquiringaccountholder
        from transactions_new as tran, eutl_accountholders as acc, eutl_accholderclassification as class
        where tran.acquiringaccountholder = acc.holdername
        and acc.rawcode=class.holder
        and TransactionDate LIKE '{month}%'"""
        sql01 = f"""select tran.transferringaccountholder
        from transactions_new as tran, eutl_accountholders as acc, eutl_accholderclassification as class
        where tran.transferringaccountholder = acc.holdername
        and acc.rawcode=class.holder
        and TransactionDate LIKE '{month}%'"""
        cursor.execute(sql00)
        acqall = cursor.fetchall()
        cursor.execute(sql01)
        transall = cursor.fetchall()
        duplall = acqall + transall
        sql= f"""select tran.acquiringaccountholder
        from transactions_new as tran, eutl_accountholders as acc, eutl_accholderclassification as class
        where tran.acquiringaccountholder = acc.holdername
        and acc.rawcode=class.holder
        and class.sector='20'
        and TransactionDate LIKE '{month}%'"""
        sql1 = f"""select tran.transferringaccountholder
        from transactions_new as tran, eutl_accountholders as acc, eutl_accholderclassification as class
        where tran.transferringaccountholder = acc.holdername
        and acc.rawcode=class.holder
        and class.sector='20'
        and TransactionDate LIKE '{month}%'"""
        cursor.execute(sql)
        acq = cursor.fetchall()
        cursor.execute(sql1)
        trans = cursor.fetchall()
        dupllist= acq+trans
        comby.append(len(get_unique(dupllist))*100/len(get_unique(duplall)))
        #print(len(acq),len(trans),len(get_unique(dupllist)))
    sec[1].plot(combx, comby)

def moneyde():#τι ποσοστό όλων των κόμβων ανά μήνα είναι και financial και γερμανικοί
    refer = ('2014-01-', '2014-02-', '2014-03-', '2014-04-', '2014-05-', '2014-06-', '2014-07-', '2014-08-', '2014-09-',
             '2014-10-', '2014-11-', '2014-12-')
    germanx = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    germany = []
    for month in refer:
        #sql = "select distinct tran.AcquiringAccountHolder from Transactions_New as tran inner join EUTL_AccountHolders as acc on tran.AcquiringAccountHolder=acc.holderName where acc.country='DE' inner join EUTL_AccHolderClassification as class on acc.rawCode=class.holder where class.category='financial'"
        sql00 = f"""select distinct tran.acquiringaccountholder
                from transactions_new as tran, eutl_accountholders as acc, eutl_accholderclassification as class
                where tran.acquiringaccountholder = acc.holdername
                and acc.rawcode=class.holder
                and TransactionDate LIKE '{month}%'"""
        sql01 = f"""select distinct tran.transferringaccountholder
                from transactions_new as tran, eutl_accountholders as acc, eutl_accholderclassification as class
                where tran.transferringaccountholder = acc.holdername
                and acc.rawcode=class.holder
                and TransactionDate LIKE '{month}%'"""
        cursor.execute(sql00)
        acqall = cursor.fetchall()
        cursor.execute(sql01)
        transall = cursor.fetchall()
        duplall = acqall + transall
        sql2= f"""select distinct tran.acquiringaccountholder
        from transactions_new as tran, eutl_accountholders as acc, eutl_accholderclassification as class
        where tran.acquiringaccountholder = acc.holdername
        and acc.country='DE'
        and acc.rawcode=class.holder
        and category='financial'
        and TransactionDate LIKE '{month}%'"""
        sql3 = f"""select distinct tran.transferringaccountholder
        from transactions_new as tran, eutl_accountholders as acc, eutl_accholderclassification as class
        where tran.transferringaccountholder = acc.holdername
        and acc.country='DE'
        and acc.rawcode=class.holder
        and category='financial'
        and TransactionDate LIKE '{month}%'"""
        cursor.execute(sql2)
        acq = cursor.fetchall()
        cursor.execute(sql3)
        trans = cursor.fetchall()
        #sql1= ("tran.AcquiringAccountHolder")
        #cursor.execute(sql)
        #result = cursor.fetchall()
        dupllist= acq+trans
        germany.append(len(get_unique(dupllist))*100/len(get_unique(duplall)))
        #print(len(acq),len(trans),len(get_unique(dupllist)))
    axs[1, 0].plot(germanx, germany, 'tab:green')
    axs[1, 0].set_title('Germany')

def moneyfr():#τι ποσοστό όλων των κόμβων ανά μήνα είναι και financial και γαλλικοί
    refer = ('2014-01-', '2014-02-', '2014-03-', '2014-04-', '2014-05-', '2014-06-', '2014-07-', '2014-08-', '2014-09-',
             '2014-10-', '2014-11-', '2014-12-')
    frenchx = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    frenchy = []
    for month in refer:
        #sql = "select distinct tran.AcquiringAccountHolder from Transactions_New as tran inner join EUTL_AccountHolders as acc on tran.AcquiringAccountHolder=acc.holderName where acc.country='DE' inner join EUTL_AccHolderClassification as class on acc.rawCode=class.holder where class.category='financial'"
        sql00 = f"""select distinct tran.acquiringaccountholder
                from transactions_new as tran, eutl_accountholders as acc, eutl_accholderclassification as class
                where tran.acquiringaccountholder = acc.holdername
                and acc.rawcode=class.holder
                and TransactionDate LIKE '{month}%'"""
        sql01 = f"""select distinct tran.transferringaccountholder
                from transactions_new as tran, eutl_accountholders as acc, eutl_accholderclassification as class
                where tran.transferringaccountholder = acc.holdername
                and acc.rawcode=class.holder
                and TransactionDate LIKE '{month}%'"""
        cursor.execute(sql00)
        acqall = cursor.fetchall()
        cursor.execute(sql01)
        transall = cursor.fetchall()
        duplall = acqall + transall
        sql2= f"""select distinct tran.acquiringaccountholder
        from transactions_new as tran, eutl_accountholders as acc, eutl_accholderclassification as class
        where tran.acquiringaccountholder = acc.holdername
        and acc.country='FR'
        and acc.rawcode=class.holder
        and category='financial'
        and TransactionDate LIKE '{month}%'"""
        sql3 = f"""select distinct tran.transferringaccountholder
        from transactions_new as tran, eutl_accountholders as acc, eutl_accholderclassification as class
        where tran.transferringaccountholder = acc.holdername
        and acc.country='FR'
        and acc.rawcode=class.holder
        and category='financial'
        and TransactionDate LIKE '{month}%'"""
        cursor.execute(sql2)
        acq = cursor.fetchall()
        cursor.execute(sql3)
        trans = cursor.fetchall()
        #sql1= ("tran.AcquiringAccountHolder")
        #cursor.execute(sql)
        #result = cursor.fetchall()
        dupllist= acq+trans
        frenchy.append(len(get_unique(dupllist))*100/len(get_unique(duplall)))
        #print(len(acq),len(trans),len(get_unique(dupllist)))
    axs[0, 1].plot(frenchx, frenchy)
    axs[0, 1].set_title('French')

def moneyuk():#τι ποσοστό όλων των κόμβων ανά μήνα είναι και financial και βρετανικοί
    refer = ('2014-01-', '2014-02-', '2014-03-', '2014-04-', '2014-05-', '2014-06-', '2014-07-', '2014-08-', '2014-09-',
             '2014-10-', '2014-11-', '2014-12-')
    ukx = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    uky = []
    for month in refer:
        #sql = "select distinct tran.AcquiringAccountHolder from Transactions_New as tran inner join EUTL_AccountHolders as acc on tran.AcquiringAccountHolder=acc.holderName where acc.country='DE' inner join EUTL_AccHolderClassification as class on acc.rawCode=class.holder where class.category='financial'"
        sql00 = f"""select distinct tran.acquiringaccountholder
                from transactions_new as tran, eutl_accountholders as acc, eutl_accholderclassification as class
                where tran.acquiringaccountholder = acc.holdername
                and acc.rawcode=class.holder
                and TransactionDate LIKE '{month}%'"""
        sql01 = f"""select distinct tran.transferringaccountholder
                from transactions_new as tran, eutl_accountholders as acc, eutl_accholderclassification as class
                where tran.transferringaccountholder = acc.holdername
                and acc.rawcode=class.holder
                and TransactionDate LIKE '{month}%'"""
        cursor.execute(sql00)
        acqall = cursor.fetchall()
        cursor.execute(sql01)
        transall = cursor.fetchall()
        duplall = acqall + transall
        sql2= f"""select distinct tran.acquiringaccountholder
        from transactions_new as tran, eutl_accountholders as acc, eutl_accholderclassification as class
        where tran.acquiringaccountholder = acc.holdername
        and acc.country='UK'
        and acc.rawcode=class.holder
        and category='financial'
        and TransactionDate LIKE '{month}%'"""
        sql3 = f"""select distinct tran.transferringaccountholder
        from transactions_new as tran, eutl_accountholders as acc, eutl_accholderclassification as class
        where tran.transferringaccountholder = acc.holdername
        and acc.country='UK'
        and acc.rawcode=class.holder
        and category='financial'
        and TransactionDate LIKE '{month}%'"""
        cursor.execute(sql2)
        acq = cursor.fetchall()
        cursor.execute(sql3)
        trans = cursor.fetchall()
        #sql1= ("tran.AcquiringAccountHolder")
        #cursor.execute(sql)
        #result = cursor.fetchall()
        dupllist= acq+trans
        uky.append(len(get_unique(dupllist))*100/len(get_unique(duplall)))
        #print(len(acq),len(trans),len(get_unique(dupllist)))
    axs[0, 0].plot(ukx, uky, 'tab:red')
    axs[0, 0].set_title('UK')

def countries():#βγάζει σε γράφημα τις 3 χώρες που ζητήθηκαν
    moneyde()
    moneyuk()
    moneyfr()
    for ax in axs.flat:
        ax.set(xlabel='month', ylabel='% of transactions')
    #for ax in axs.flat:
    #    ax.label_outer()

def sectors():#βγάζει σε γράφημα τους 2 τομείς που ζητήθηκαν
    mineral()
    sector20()
    for el in sec.flat:
        el.set(xlabel='month', ylabel='% of transactions')
    for el in sec.flat:
        el.label_outer()

if __name__ == '__main__':
    ignite()#ξεκινάει τη βάση
    ###### Οι παρακάτω γραμμές πάνε εναλλάξ αν ζεύγη(σχόλια τα 1,3 ή τα 2,4)#####
    fig, axs = plt.subplots(2, 2)
    #fig, sec = plt.subplots(2, 1)
    countries()
    #sectors()

    plt.show()#εμφανίζει το γράφημα
