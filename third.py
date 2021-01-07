import mysql.connector
from msqlconfig import cursor, db
import datetime
import numpy as np
import matplotlib.pyplot as plt
from main import get_unique,ignite



def mineral():
    refer=('2008-12-','2009-12-','2010-12-','2011-12-','2012-12-','2013-12-','2014-12')
    mineralx = ['2008','2009','2010','2011','2012','2013','2014']
    mineraly=[]
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
        sql = f"""select distinct tran.acquiringaccountholder
            from transactions_new as tran, EUTL_Installations_orAir as inst
            where tran.acquiringaccountholder = inst.onoma
            and (inst.mainActivity='2' or inst.mainActivity='21')      
            and TransactionDate LIKE '{month}%'"""
        sql1= f"""select distinct tran.transferringaccountholder
            from transactions_new as tran, EUTL_Installations_orAir as inst
            where tran.transferringaccountholder = inst.onoma
            and (inst.mainActivity='2' or inst.mainActivity='21')      
            and TransactionDate LIKE '{month}%'"""
        cursor.execute(sql)
        acq = cursor.fetchall()
        cursor.execute(sql1)
        trans = cursor.fetchall()
        dupllist= acq+trans
        mineraly.append(len(get_unique(dupllist))*100/len(get_unique(duplall)))
        #print(len(acq),len(trans),len(get_unique(dupllist)))
    sec[0].plot(mineralx, mineraly)


def sector20():
    refer = ('2008-12-', '2009-12-', '2010-12-', '2011-12-', '2012-12-', '2013-12-', '2014-12')

    combx = ['2008','2009','2010','2011','2012','2013','2014']
    comby = []
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
        sql = f"""select distinct tran.acquiringaccountholder
            from transactions_new as tran, EUTL_Installations_orAir as inst
            where tran.acquiringaccountholder = inst.onoma
            and (inst.country='DE' or inst.country='FR' or inst.country='NL')
            and inst.mainActivity='20'       
            and TransactionDate LIKE '{month}%'"""
        sql1= f"""select distinct tran.transferringaccountholder
            from transactions_new as tran, EUTL_Installations_orAir as inst
            where tran.transferringaccountholder = inst.onoma
            and (inst.country='DE' or inst.country='FR' or inst.country='NL')
            and inst.mainActivity='20'       
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
    refer = ('2008-12-', '2009-12-', '2010-12-', '2011-12-', '2012-12-', '2013-12-', '2014-12')

    germanx = ['08','09','10','11','12','13','14']

    germany = []
    for month in refer:
        #sql = "select distinct tran.AcquiringAccountHolder from Transactions_New as tran inner join EUTL_AccountHolders as acc on tran.AcquiringAccountHolder=acc.holderName where acc.country='DE' inner join EUTL_AccHolderClassification as class on acc.rawCode=class.holder where class.category='financial'"
        #sql0 = f"SELECT * FROM Transactions_New WHERE TransactionDate LIKE '{month}%'"
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
    axs[1,0].plot(germanx, germany,'tab:green')
    axs[1, 0].set_title('Germany')


def moneyfr():#τι ποσοστό όλων των κόμβων ανά μήνα είναι και financial και γαλλικοί
    refer = ('2008-12-', '2009-12-', '2010-12-', '2011-12-', '2012-12-', '2013-12-', '2014-12')

    frenchx = ['08','09','10','11','12','13','14']
    frenchy = []
    for month in refer:
        #sql = "select distinct tran.AcquiringAccountHolder from Transactions_New as tran inner join EUTL_AccountHolders as acc on tran.AcquiringAccountHolder=acc.holderName where acc.country='DE' inner join EUTL_AccHolderClassification as class on acc.rawCode=class.holder where class.category='financial'"
        #sql0 = f"SELECT * FROM Transactions_New WHERE TransactionDate LIKE '{month}%'"
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
    axs[0,1].plot(frenchx, frenchy)
    axs[0, 1].set_title('French')


def moneyuk():#τι ποσοστό όλων των κόμβων ανά μήνα είναι και financial και βρετανικοί
    refer = ('2008-12-', '2009-12-', '2010-12-', '2011-12-', '2012-12-', '2013-12-', '2014-12')

    ukx = ['08','09','10','11','12','13','14']
    uky = []
    for month in refer:
        #sql = "select distinct tran.AcquiringAccountHolder from Transactions_New as tran inner join EUTL_AccountHolders as acc on tran.AcquiringAccountHolder=acc.holderName where acc.country='DE' inner join EUTL_AccHolderClassification as class on acc.rawCode=class.holder where class.category='financial'"
        #sql0 = f"SELECT * FROM Transactions_New WHERE TransactionDate LIKE '{month}%'"
        sql00=f"""select distinct tran.acquiringaccountholder
        from transactions_new as tran, eutl_accountholders as acc, eutl_accholderclassification as class
        where tran.acquiringaccountholder = acc.holdername
        and acc.rawcode=class.holder
        and TransactionDate LIKE '{month}%'"""
        sql01=f"""select distinct tran.transferringaccountholder
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
    axs[0,0].plot(ukx, uky,'tab:red')
    axs[0, 0].set_title('UK')

def countries():
    moneyde()
    moneyuk()
    moneyfr()
    for ax in axs.flat:
        ax.set(xlabel='year', ylabel='% of transactions')
    for ax in axs.flat:
        ax.label_outer()

def sectors():
    mineral()
    sector20()
    for el in sec.flat:
        el.set(xlabel='year', ylabel='% of transactions')
    for el in sec.flat:
        el.label_outer()


ignite()
###### Τα παρακάτω πάνε εναλλάξ αν ζεύγη#####
fig, axs = plt.subplots(2, 2)
#   fig, sec = plt.subplots(2, 1)
countries()
#sectors()

plt.show()

