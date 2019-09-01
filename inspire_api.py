#!/usr/bin/python3

################################################################################
# This script is meant to retrieve bibliometric data from inspirehep.net
################################################################################

# Import useful packages
import requests
import operator
import time

# Send a search query to inspirehep.net
def inspire_search(query,ot=False):
    url = 'https://inspirehep.net/search?p={}&of=recjson'\
            .format(query)
    if ot:
        url += '&ot={}'.format(ot)
    url += '&rg=25000'
    page = requests.get(url)
    data = page.json()
    return data

# Get the set of authors who cite a given author, all identified by their author ids
def unique_citing_authors(author_id):
    data = inspire_search('refersto:author:{}'.format(author_id),'authors')
    names = set([d2['full_name'] for d in data for d2 in d['authors']])
    #print(len(names))
    return names

# Get the number of authors on a certain work
def NAuthors(recid):
    data = inspire_search('recid:{}'.format(recid),'authors')
    auth_list = [auth for d in data for auth in d['authors']]
    return(len(auth_list))

################################################################################
# Below I attempt to implement the citation coin metric of arxiv:1803.10713
################################################################################

def NicitP(recid):
    data = inspire_search('refersto:recid:{}'.format(recid),'reference')
    refs = filter(None,[d['reference'] for d in data])
    nlist = [(1/len(r)) for r in refs]
    return(sum(nlist))

# This is what 1803.10713 refers to as citation coin
def AuthCoin(author_id):
    data = inspire_search('author:{}'.format(author_id),'recid')
    recids = [d['recid'] for d in data]
    tosum = [(NicitP(r) - 1)/NAuthors(r) for r in recids]
    return sum(tosum)

# Given an author, list the AutCoin of each paper
def AuthCoinByPaper(author_id):
    data = inspire_search('author:{}'.format(author_id),'title,recid')
    papers = [(d['title']['title'],d['recid']) for d in data]
    ret = sorted([{'title':p[0], 'recid':p[1], 'Nauthors':NAuthors(p[1]), 'authcoin':(NicitP(p[1]) - 1)} for p in papers], key=lambda d: d['authcoin'], reverse=True)
    return ret

# The AuthCoin of only an authors top n papers
def TopPapersAuthCoin(author_id, n):
    plist = AuthCoinByPaper(author_id)
    clist = sorted([p['authcoin']/p['Nauthors'] for p in plist], reverse=True)
    if len(clist)>n:
        clist = clist[:n]
    return sum(clist)

# Sum up authcoin of papers, restricting to those that have positive authcoin.
def PositiveAuthCoin(author_id):
    plist = AuthCoinByPaper(author_id)
    clist = list(filter(lambda x: x>0, [p['authcoin']/p['Nauthors'] for p in plist]))

    return sum(clist)

# Actually do stuff
def main():
    print("\nHello World\n\n")

    # jrefs = unique_citing_authors('J.Couch.1')
    # srefs = unique_citing_authors('S.Eccles.1')
    # prefs = unique_citing_authors('Phuc.H.Nguyen.1')
    # arefs = unique_citing_authors('A.R.Brown.1')
    # brefs = unique_citing_authors('B.Swingle.1')
    #
    # many_authors = jrefs.union(prefs).union(arefs).union(brefs).union(srefs)
    # print(len(jrefs))
    # print(len(srefs))
    # print(len(prefs))
    # print(len(arefs))
    # print(len(brefs))
    # print(len(many_authors))


    TGProfs = ['E.Caceres.1','J.Distler.1','W.Fischler.1','V.S.Kaplunovsky.1','Can.Kilic.1','R.A.Matzner.1','S.Paban.1','A.B.Zimmerman.1']
    OtherPeople = ['J.Couch.1','H.Marrochio.1','Ming.Lei.Xiao.1','Phuc.H.Nguyen.1','J.F.Pedraza.1']

    #print(NicitP('451647'))
    #print(NAuthors('1681268'))
    papers = AuthCoinByPaper('J.Couch.1')
    print(papers)
    for p in OtherPeople:
        print("id:{}, authcoin:{}".format(p,AuthCoin(p)))
        time.sleep(0.5)
    print()

if __name__ == '__main__':
	main()
