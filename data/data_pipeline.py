import gzip
import pandas as pd
import numpy as np

def generate_dfs(file_idx):

    with gzip.open('listings.csv.gz', 'rb') as l:
        ls = pd.read_csv(l)

    with gzip.open('reviews.csv.gz', 'rb') as r:
        rv = pd.read_csv(r)

    for idx in xrange(file_idx):
        if idx > 0:
            l_name = 'listings ({}).csv.gz'.format(str(idx))
            r_name = 'reviews ({}).csv.gz'.format(str(idx))

            with gzip.open(l_name, 'rb') as lst:
                ls=pd.concat([ls, pd.read_csv(lst)], ignore_index=True)

            with gzip.open(r_name, 'rb') as rvs:
                rv=pd.concat([rv, pd.read_csv(rvs)], ignore_index=True)

            print str(idx)

    return ls, rv

def write_to_csv(df_ls,df_rv):

    df_ls.to_csv('listings.csv')
    df_rv.to_csv('reviews.csv')


if __name__ == '__main__':
    listings, reviews = generate_dfs(43)
    write_to_csv(listings, reviews)

    l=pd.read_csv('listings.csv')
    r=pd.read_csv('reviews.csv')
