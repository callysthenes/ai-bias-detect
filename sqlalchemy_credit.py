#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 06:42:53 2022

@author: stephaniegessler
"""



import pandas as pd
df = pd.read_csv("synthetic_credit_card_approval.csv")
df = df.fillna(0)

# #CREATING A CONNECTION WITH THE DATABASE ENGINE, IT CAN BE ANY SQL Server, MySql, Posgres, etc...
from sqlalchemy import create_engine
from sqlalchemy.exc import IntegrityError
engine = create_engine('sqlite:///./synthetic_credit_card.db')

# DECLARING THE OBJECT - Object-relational mapping
from sqlalchemy.ext.declarative import declarative_base
Base = declarative_base()

# EXTENDING THE Base OBJECT INTO AN OBJECT CALLED User
from sqlalchemy import Column, Integer
class CreditCard(Base):
    __tablename__ = 'credit_card'
    id = Column(Integer, primary_key=True)
    children= Column(Integer) 
    group = Column(Integer)
    income = Column(Integer)
    own_car	= Column(Integer)
    own_housing = Column(Integer)
    target = Column(Integer)



# WE RUN create_all SO THAT IF THE TABLE DOES NOT EXIST IT CREATES IT
Base.metadata.create_all(engine)

# WE CREATE A SESSION TO BE ABLE TO INSERT, UPDATE AND DELETE DATA.
from sqlalchemy.orm import sessionmaker
DBSession = sessionmaker(bind=engine)
session = DBSession()



# INSERTING 1000 USERS EACH TIME
for i in range(len(df)):
    credit = CreditCard(children=int(df['Num_Children'].iloc[i]),
                           group=int(df['Group'].iloc[i]),
                           income=int(df['Income'].iloc[i]),
                           own_car=int(df['Own_Car'].iloc[i]),
                           own_housing=int(df['Own_Housing'].iloc[i]),
                           target=int(df['Target'].iloc[i]))
            
    session.add(credit)

#   session.commit()  
    if ( i % 1000 == 0):
        print ("inserting 1000 credit - i = {} - name last company inserted = {}".format(i,df['Target'].iloc[i]))
    try:    
        session.commit()
    except IntegrityError:
        session.rollback()


session.commit()