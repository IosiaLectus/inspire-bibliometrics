#!/usr/bin/python3

################################################################################
# This script is meant to retrieve bibliometric data from inspirehep.net
################################################################################

# Import useful packages
import requests
import operator
import time

# Send a search query to inspirehep.net
def inspire_search(query, ot=False, verbose=False, start_at=0):
    url = 'https://inspirehep.net/search?p={}&of=recjson'\
            .format(query)
    if ot:
        url += '&ot={}'.format(ot)
    url += '&rg=250'
    if start_at:
        url += '&jrec={}'.format(start_at)
    page = requests.get(url)
    data = page.json()
    if verbose:
        print("\n"+url+"\n")
    return data

# Retrieve record from inspirehep.net by recid
def inspire_record(recid, ot=False, verbose=False):
    url = 'https://inspirehep.net/record/{}?of=recjson'\
            .format(recid)
    if ot:
        url += '&ot={}'.format(ot)
    page = requests.get(url)
    data = page.json()
    if verbose:
        print("\n"+url+"\n")
    return data

# Get the set of authors who cite a given author, all identified by their author ids
def unique_citing_authors(author_id):
    data = inspire_search('refersto:author:{}'.format(author_id),'authors')
    names = set([d2['full_name'] for d in data for d2 in d['authors']])
    #print(len(names))
    return names

# Get the number of authors on a certain work
def NAuthors(recid):
    data = inspire_record(recid,'authors')
    auth_list = [auth for d in data for auth in d['authors']]
    return(len(auth_list))

# Get number of citations from a paper
def NCitations(recid):
    #data = inspire_search('refersto:recid:{}'.format(recid),'reference')
    #return len(data)
    record = inspire_record(recid,"number_of_citations")
    if len(record)<1:
        return 0
    return record[0]["number_of_citations"]

# Get citations for each paper by an author
def CitationsByPaper(author_id):
    data = inspire_search('author:{}'.format(author_id),'title,recid')
    papers = [(d['title']['title'],d['recid']) for d in data]
    ret = sorted([{'title':p[0], 'recid':p[1], 'citations':NCitations(p[1])} for p in papers], key=lambda d: d['citations'], reverse=True)
    return ret

# Get recid by title
def recid_from_title(title):
    data = inspire_search("title:'{}'".format(title),'recid')
    if len(data)<1:
        return -1
    return data[0]['recid']

# Get paper abstract from recid
def get_abstract(recid):
    record = inspire_record(recid,"abstract")
    # Get abstract text from json
    pre_abstract = record[0]['abstract']
    if type(pre_abstract) is dict:
        return pre_abstract['summary']
    return record[0]['abstract'][0]['summary']

# Get citing papers
def get_citing_papers(recid, max_iterations=5, keylist=[], verbose=False):
    # parse keylist into a string to add to the request
    keystring = parse_keylist(['recid']+keylist)
    # maker request
    data0 = inspire_search('refersto:recid:{}'.format(recid), keystring, verbose)
    data = data0
    iteration_count = 0
    while len(data0)==250 and iteration_count < max_iterations:
        iteration_count = iteration_count + 1
        if verbose:
            print("Iteration count for recid {}: {}".format(recid, iteration_count))
        data0 = inspire_search('refersto:recid:{}'.format(recid), keystring, verbose, 250*iteration_count)
        data = data + data0
    return data


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

################################################################################
# A few things I want to define myself
################################################################################

# Get citations which each in turn have at least 10 citations (or go deeper with n>1)
# In more detail, for n=0 simply return the citation count
# For n=1, return the number of citations such that the citing paper has at least k citations.
# For n>1, return the number of citations such that the citing paper has at an i10(n-1,k) score of at least k.
# This is inspired by the i10 score defined for authors
def i10_citations(recid,n,k=10):
    if n<1:
        return NCitations(recid)
    data = inspire_search('refersto:recid:{}'.format(recid),'recid,number_of_citations')
    data = [d for d in data if d]
    print(data)
    print()
    data2 = []
    if n==1:
        data2 = [d['number_of_citations'] for d in data]
    else:
        data2 = [i10_citations(d['recid'],n-1,k) for d in data]
    print(data2)
    print()
    data2 = [x for x in data2 if x>k]
    return len(data2)

# Just as the above is a recursive i10 score for papers, here we define a recursive h-index for papers.
def h_index_citations(recid,n):
    if n<1:
        return NCitations(recid)
    data = inspire_search('refersto:recid:{}'.format(recid),'recid,number_of_citations')
    data = [d for d in data if d]
    print(data)
    print()
    data2 = []
    if n==1:
        data2 = [d['number_of_citations'] for d in data]
    else:
        data2 = [h_index_citations(d['recid'],n-1) for d in data]
    print(data2)
    print()
    h_index = 0
    while len(data2)>h_index:
        h_index += 1
        data2 = [x for x in data2 if x>h_index]
    return h_index

# Parse a list of keys into a string that can be included in a request
def parse_keylist(keylist):
    ret = ''
    if len(keylist)<1:
        return ret
    for key in keylist[:-1]:
        ret += key
        ret += ','
    ret += keylist[-1]
    return ret

# get all papers 'downstream' from the current paper in the citation graph, i.e. the paper's descendants.
def get_descendants(recid, max_number=10**4, keylist=[], verbose=False):
    # parse keylist into a string to add to the request
    keystring = parse_keylist(['recid']+keylist)
    # maker request
    data = get_citing_papers(recid, max_number, keylist, verbose)
    # save results in 'data'
    data = [d for d in data if d]
    i = 0
    # print things for debugging
    if verbose:
        print()
        print("max_number is {}".format(max_number))
        print()
    # Iteratively add descendants, until there is nothing to add or we reach the maximum number.
    while i < len(data) and i < max_number:
        this_paper = data[i]
        if isinstance(this_paper, dict):
            data.extend(get_citing_papers(data[i]['recid'], max_number, keylist, verbose))
        # solution from https://www.geeksforgeeks.org/python-removing-duplicate-dicts-in-list/ to remove duplicates
        data = [j for n, j in enumerate(data) if not j in data[n+1:]]
        data = [d for d in data if d]
        # print things for debugging
        if verbose:
            print("Iteration number: {}".format(i))
            print ("Length of list: {}".format(len(data)))
            print()
        i += 1
    data = sorted(data, key=lambda x: x['recid'])
    return data

# get total number of descendants.
def number_of_descendants(recid):
    return len(get_descendants(recid))

# Get detailed records from list of recids
#def get_records(recid_list, keylist=[])


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
    OtherPeople = ['Josiah.D.Couch.1','H.Marrochio.1','Ming.Lei.Xiao.1','Phuc.H.Nguyen.1','J.F.Pedraza.1']

    print()
    #print(recid_from_title("Noether charge, black hole volume, and complexity"))
    #print()
    #print(inspire_record('1681268'))
    #recid = recid_from_title("Noether charge, black hole volume, and complexity")
    recid = 451647
    descendant_list = get_descendants(recid, 10**5, [], True)
    print(len(descendant_list))
    print()
    #print(descendant_list)
    #print()
    filestring = ""
    for dsc in descendant_list:
        #print(get_abstract(dsc['recid']))
        filestring += str(dsc['recid'])
        filestring += "\n"
        #print()
    my_file = open('recid_list.txt','w')
    my_file.write(filestring)
    my_file.close()

    #print(NicitP('451647'))
    #print(NAuthors('1681268'))
    #papers = AuthCoinByPaper('Josiah.D.Couch.1')
    #print(papers)
    # for p in OtherPeople:
    #    print("id:{}, authcoin:{}".format(p,AuthCoin(p)))
    #    time.sleep(0.5)
    # print()

if __name__ == '__main__':
	main()
