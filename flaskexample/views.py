#load Flask
from flaskexample import app
from flask import Flask, request, render_template
#load Python packages
import pickle
import numpy as np
import pandas as pd
from gensim.parsing.preprocessing import preprocess_documents #used to parse data
from sklearn.feature_extraction.text import TfidfVectorizer #used to parse data
from sklearn.metrics.pairwise import linear_kernel #used to parse data
#load Python modules
import math
import random
import re
#load definitions from model file
from .model.a_Model import load_csv
from .model.a_Model import parse_df
from .model.a_Model import cos_sim
from .model.a_Model import weights
from .model.a_Model import w_sim
from .model.a_Model import scores
from .model.a_Model import titles
from .model.a_Model import results
from .model.a_Model import rand_indices
from .model.a_Model import rand_titles
from .model.a_Model import rand_authors
from .model.a_Model import rand_urls
from .model.a_Model import rand_covers
from .model.a_Model import book_ratings
from .model.a_Model import cover_results
from .model.a_Model import title_results
from .model.a_Model import authors_results
from .model.a_Model import url_results
from .model.a_Model import rating_results
from .model.a_Model import nratings_results
from .model.a_Model import mostsim_results
from .model.a_Model import msidx
from .model.a_Model import mostsimurl_results
from .model.a_Model import blurb_results
from .model.a_Model import final_results

#load model results
#df_rec = load_csv("file:///home/ubuntu/NewBook/flaskexample/prek_df.csv")
df_rec = load_csv("file:///home/john/Documents/prek/flaskexample/prek_df.csv")
df_rec = parse_df(df_rec)
sim = cos_sim(df_rec)
coeffs = weights(df_rec)
wsim = w_sim(coeffs,sim)

@app.route('/')
@app.route('/input')
def input():
    l = rand_indices() #numpy array of 30 random numbers; numbers determine which books are displayed on input page
    rtitles = rand_titles(l,df_rec) #titles corresponding to random numbers
    rauthors = rand_authors(l,df_rec) #authors corresponding to random numbers
    rurls = rand_urls(l,df_rec) #Goodreads urls corresponding to random numbers
    rcovers = rand_covers(l,df_rec) #book cover urls corresponding to random numbers
    return render_template('input.html',rtitles=rtitles,rauthors=rauthors,rurls=rurls,rcovers=rcovers,rlist=l)

@app.route('/output')
def output():
    #get value of all user ratings, r1-r30
    r1 = request.args.get('rate1')
    r1 = '0' if r1 is None else r1
    r2 = request.args.get('rate2')
    r2 = '0' if r2 is None else r2
    r3 = request.args.get('rate3')
    r3 = '0' if r3 is None else r3
    r4 = request.args.get('rate4')
    r4 = '0' if r4 is None else r4
    r5 = request.args.get('rate5')
    r5 = '0' if r5 is None else r5
    r6 = request.args.get('rate6')
    r6 = '0' if r6 is None else r6
    r7 = request.args.get('rate7')
    r7 = '0' if r7 is None else r7
    r8 = request.args.get('rate8')
    r8 = '0' if r8 is None else r8
    r9 = request.args.get('rate9')
    r9 = '0' if r9 is None else r9
    r10 = request.args.get('rate10')
    r10 = '0' if r10 is None else r10
    r11 = request.args.get('rate11')
    r11 = '0' if r11 is None else r11
    r12 = request.args.get('rate12')
    r12 = '0' if r12 is None else r12
    r13 = request.args.get('rate13')
    r13 = '0' if r13 is None else r13
    r14 = request.args.get('rate14')
    r14 = '0' if r14 is None else r14
    r15 = request.args.get('rate15')
    r15 = '0' if r15 is None else r15
    r16 = request.args.get('rate16')
    r16 = '0' if r16 is None else r16
    r17 = request.args.get('rate17')
    r17 = '0' if r17 is None else r17
    r18 = request.args.get('rate18')
    r18 = '0' if r18 is None else r18
    r19 = request.args.get('rate19')
    r19 = '0' if r19 is None else r19
    r20 = request.args.get('rate20')
    r20 = '0' if r20 is None else r20
    r21 = request.args.get('rate21')
    r21 = '0' if r21 is None else r21
    r22 = request.args.get('rate22')
    r22 = '0' if r22 is None else r22
    r23 = request.args.get('rate23')
    r23 = '0' if r23 is None else r23
    r24 = request.args.get('rate24')
    r24 = '0' if r24 is None else r24
    r25 = request.args.get('rate25')
    r25 = '0' if r25 is None else r25
    r26 = request.args.get('rate26')
    r26 = '0' if r26 is None else r26
    r27 = request.args.get('rate27')
    r27 = '0' if r27 is None else r27
    r28 = request.args.get('rate28')
    r28 = '0' if r28 is None else r28
    r29 = request.args.get('rate29')
    r29 = '0' if r29 is None else r29
    r30 = request.args.get('rate30')
    r30 = '0' if r30 is None else r30
    #make a list of all user ratings
    rtgs = [r1,r2,r3,r4,r5,r6,r7,r8,r9,r10,r11,r12,r13,r14,r15,r16,r17,r18,r19,r20,r21,r22,r23,r24,r25,r26,r27,r28,r29,r30]
    bklist = str(request.args.get('bklist')) #indices corresponding to user ratings r1-r30
    l  = book_ratings(bklist,rtgs) #numpy array of user ratings, padded with zeros
    global wsim #grab pre-loaded model data
    scrs = scores(wsim,l) #adjust model data for user ratings
    ttles = titles(wsim,l,df_rec) #grabbing classic book most similar to each lesser-known book
    urlz = msidx(wsim,l,df_rec) #grabbing Goodreads url of classic book most similar to lesser-known book
    rslts0 = results(scrs, df_rec, ttles,urlz) #constructing dataframe with results for each lesser-known book
    rslts = final_results(df_rec,l,rslts0) #re-organizing dataframe to account for diversity of user preferences
    titleres = title_results(rslts) #grabbing title column from results dataframe
    authorsres = authors_results(rslts) #grabbing authors column from results dataframe
    coverres = cover_results(rslts) #grabbing cover column
    urlres = url_results(rslts) #grabbing Goodreads url column
    nratingsres = nratings_results(rslts) #grabbing number of user ratings column
    ratingres = rating_results(rslts)  #grabbing average rating column
    mostsimres = mostsim_results(rslts) #grabbing 'most similar to' column
    mostsimurlres = mostsimurl_results(rslts) #grabbing "url of 'most similar to'" column
    blurbres = blurb_results(rslts) #grabbing blurb column             
    return render_template('output.html',titleres=titleres,authorsres=authorsres,coverres=coverres,urlres=urlres,nratingsres=nratingsres,ratingres=ratingres,mostsimres=mostsimres,mostsimurlres=mostsimurlres,blurbres=blurbres)

