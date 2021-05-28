import mysql.connector
from mysqlconfig import cursor

DB_NAME = 'whateveryoulike'

def create_database():
    cursor.execute("CREATE DATABASE IF NOT EXISTS {} DEFAULT CHARACTER SET 'utf8'".format(DB_NAME))
    print("Database {} created".format((DB_NAME)))