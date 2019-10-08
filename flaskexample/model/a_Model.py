#load Python modules and packages
import pandas as pd
import numpy as np
import math
import random
import re
from gensim.parsing.preprocessing import preprocess_documents
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

#definitions below are discussed in views.py
def load_csv(filename):
    df = pd.read_csv(filename)
    return df

def parse_df(df):
    df['parsed series'] = preprocess_documents(df['series'].astype('str'))
    df['parsed reviews'] = preprocess_documents(df['reviews'].astype('str'))
    df['parsed blurb'] = preprocess_documents(df['blurb'].astype('str'))
    df['parsed'] = df['parsed reviews']+df['parsed blurb']+df['parsed series']
    df = df.sort_values(by=['nratings'],ascending=False).reset_index(drop=True)
    return df

def cos_sim(df):
    prsd = df['parsed'].astype('str')
    prsd2 = np.array([])
    for i in range(len(prsd)):
        lst = prsd[i].split(',')
        lstys = ""
        for j in range(len(lst)):
            lsty = lst[j].replace("]","").replace("[","")[2:-1]
            lsty = lsty.replace("'","").replace("-","")
            lstys = lsty + " " + lstys
        prsd2 = np.append(lstys,prsd2)
    tfidf = TfidfVectorizer().fit_transform(prsd2)
    sim_mtx = np.array([])
    for j in range(704):
        cos_sims = linear_kernel(tfidf[j:j+1], tfidf[705:1220]).flatten()
        sim_mtx = np.append(sim_mtx, cos_sims)
    sim_mtx = np.split(sim_mtx,704)
    return sim_mtx

def weights(df):
    df_rev = df.sort_values(by=['nratings']).reset_index(drop=True)
    cjs = np.array([])
    for j in range(704):
        try:
            pub = df_rev['pub_year'][j]
            rat = df_rev['rating'][j]
            tj = 1+ 0.2/35*(np.heaviside(1985,1)*(pub-1985) + (1-np.heaviside(1985,1))*(1985-pub) - (1-np.heaviside(1950,1))*(1985-pub))
            rj = 1+ 0.2 * np.exp(rat-5)
            if math.isnan(pub) == False:
                cj = rj*tj
            else:
                cj = rj
            cjs = np.append(cjs,cj)
        except:
            cjs = np.append(cjs,rj)
    return cjs

def w_sim(wts,sim):
    uvecs = np.array([])
    for j in range(len(sim)):
        uvec = wts[j]*sim[j]
        uvecs = np.append(uvecs,uvec)
    uvecs = np.split(uvecs,704)
    return uvecs

def scores(wsim,user_input):
    scores = np.array([])
    user_input = np.where(user_input==5, 15, user_input)
    user_input = np.where(user_input==4, 4, user_input)
    user_input = np.where(user_input==3, 2, user_input)
    user_input = np.where(user_input==2, 0, user_input)
    user_input = np.where(user_input==1, 0, user_input)
    for j in range(len(wsim)):
        score = np.dot(user_input,wsim[j])
        scores = np.append(scores,score)
    return scores

def titles(wsim,user_input,df):
    df_rev = df.sort_values(by=['nratings']).reset_index(drop=True)
    ttls = np.array([])
    user_input = np.where(user_input==5, 15, user_input)
    user_input = np.where(user_input==4, 4, user_input)
    user_input = np.where(user_input==3, 2, user_input)
    user_input = np.where(user_input==2, 0, user_input)
    user_input = np.where(user_input==1, 0, user_input)
    for j in range(704):
        weights = np.multiply(user_input,wsim[j])
        midx = np.argmax(weights)
        ttl = df_rev['title'][705+midx]
        ttls = np.append(ttls,ttl)
    return ttls

def msidx(wsim,user_input,df):
    df_rev = df.sort_values(by=['nratings']).reset_index(drop=True)
    ttls = np.array([])
    user_input = np.where(user_input==5, 15, user_input)
    user_input = np.where(user_input==4, 4, user_input)
    user_input = np.where(user_input==3, 2, user_input)
    user_input = np.where(user_input==2, 0, user_input)
    user_input = np.where(user_input==1, 0, user_input)
    for j in range(704):
        weights = np.multiply(user_input,wsim[j])
        midx = np.argmax(weights)
        ttl = df_rev['url'][705+midx]
        ttls = np.append(ttls,ttl)
    return ttls

def results(scores,df,titles,urls):
    df_rev = df.sort_values(by=['nratings']).reset_index(drop=True)
    df_score = pd.DataFrame(columns=['title','authors','rating', 'nratings','cover','url','score','most similar to', 'most sim url'])
    df_score['title'] = df_rev['title'][0:len(scores)]
    df_score['authors'] = df_rev['authors'][0:len(scores)]
    df_score['rating'] = df_rev['rating'][0:len(scores)]
    df_score['nratings'] = df_rev['nratings'][0:len(scores)]
    df_score['cover'] = df_rev['cover'][0:len(scores)]
    df_score['url'] = df_rev['url'][0:len(scores)]
    df_score['blurb'] = df_rev['blurb'][0:len(scores)]
    df_score['score'] = scores
    df_score['most similar to'] = titles
    df_score['most sim url'] = urls
    df_score = df_score.sort_values(by=['score'],ascending=False).reset_index(drop=True)
    return df_score.iloc[0:10]

def cover_results(df_res):
    return df_res['cover']

def url_results(df_res):
    return df_res['url']

def title_results(df_res):
    return df_res['title']

def authors_results(df_res):
    for j in range(len(df_res)):
        df_res['authors'][j] = df_res['authors'][j][2:-2].replace("'","")
    return df_res['authors']

def rating_results(df_res):
    return df_res['rating']

def nratings_results(df_res):
    return df_res['nratings']

def mostsim_results(df_res):
    return df_res['most similar to']

def mostsimurl_results(df_res):
    return df_res['most sim url']

def blurb_results(df_res):
    for j in range(len(df_res)):
        df_res['blurb'][j] = df_res['blurb'][j].splitlines()[1] + ' [...]'
    return df_res['blurb']


def rand_indices():
    nrats = 30
    l = 705 + np.random.choice(515, size=nrats, replace = False)
    l = np.asarray(l)
    return l

def rand_titles(l,df):
    df_rev = df.sort_values(by=['nratings']).reset_index(drop=True)
    rands = np.array([])
    for j in range(len(l)):
        ix = l[j]
        rand = df_rev['title'][ix]
        rands = np.append(rands,rand)
    return rands

def rand_authors(l,df):
    df_rev = df.sort_values(by=['nratings']).reset_index(drop=True)
    rands = np.array([])
    for j in range(len(l)):
        ix = l[j]
        rand = df_rev['authors'][ix][2:-2].replace("'","")
        rands = np.append(rands,rand)
    return rands

def rand_urls(l,df):
    df_rev = df.sort_values(by=['nratings']).reset_index(drop=True)
    rands = np.array([])
    for j in range(len(l)):
        ix = l[j]
        rand = df_rev['url'][ix]
        rands = np.append(rands,rand)
    return rands

def rand_covers(l,df):
    df_rev = df.sort_values(by=['nratings']).reset_index(drop=True)
    rands = np.array([])
    for j in range(len(l)):
        ix = l[j]
        rand = df_rev['cover'][ix]
        rands = np.append(rands,rand)
    return rands

def book_ratings(indices,ratings):
    rtgs = np.zeros(515)
    indices = re.sub(r'[^\w\s]','',indices).strip()
    indices = re.split('  | ',indices)
    for j in range(len(indices)):
        indices[j] = indices[j].replace("\r\n","")
    indices = np.asarray(indices).astype(int)
    ratings = np.asarray(ratings).astype(int)
    for j in range(len(ratings)):
        idx = indices[j]-705
        rtgs[idx] = ratings[j]
    return rtgs

def final_results(df,user_input,results):
    df_rev = df.sort_values(by=['nratings']).reset_index(drop=True)
    niques = results['most similar to'].unique()
    rgs = np.array([])
    for j in range(10):    
        try:
            idx = np.where(df_rev['title']==niques[j])[0][0]-705
            rg = user_input[idx:idx+1]
            rgs = np.append(rgs,rg)
        except:
            continue
    inds=np.array([0,1,2,3,4,5,6,7,8,9])
    for j in range(10):
        try:
            if rgs[j]>3:
                ind =results[results['most similar to']==niques[j]].first_valid_index()
                inds[j] = ind
        except:
            continue
    val = inds.astype(int).tolist()
    idx = val + results.index.drop(val).tolist()
    rslts2 = results.reindex(idx)
    rslts2 = rslts2.reset_index()
    return rslts2[0:10]


