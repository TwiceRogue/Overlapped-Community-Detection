import networkx as nx
import Learn_node_vector as n2v
import gensim.models.word2vec as w2v
from sklearn import mixture
import matplotlib
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from numpy import *
num_path = 50
length_path = 10
emb_dim = 100
winsize = 3
rw_filename = "sentences.txt"
emb_filename = "emb.txt"

if __name__ == '__main__':

    f = open("network_30.dat", 'r+')
    c = open("community_30.dat","r+")
    # G = nx.Graph()
    G = nx.read_gml("karate.gml")
    # G.add_nodes_from([1, 1000])
    # print G.number_of_nodes()
    # community = []
    # for comu in c.readlines():
    #     lab = comu.strip('\n')
    #     labe = lab.split('\t')
    #     community.append(labe[1])
    #
    # communitylabel = np.array(community)
    # # print communitylabel
    # for line in f.readlines():
    #     edge = line.strip('\n')
    #     nodes = edge.split('\t')
    #     G.add_edge(nodes[0], nodes[1], weight=1.0)
    # print G.number_of_nodes()
    for i, j in G.edges():
        G.edge[i][j]['weight'] = 1.0
    # nx.draw(G)
    # plt.show()
    model_tsne = TSNE(n_components=2, random_state=0)

    # print G.neighbors('1')
    S = n2v.build_node_alias(G)
    n2v.create_rand_walks(S, num_path, length_path, rw_filename)
    sentences = w2v.LineSentence(rw_filename)
    model_w2v = w2v.Word2Vec(sentences, size=emb_dim,window=winsize,min_count=0,sg=1,negative=5, sample=1e-1, workers=5,iter=3)
    model_w2v.save_word2vec_format(emb_filename)
    nodeslist = G.nodes()
    nodelist_int = [int(x) for x in nodeslist]
    nodelist_int_sort = sorted(nodelist_int)
    nodelist = [str(x) for x in nodelist_int_sort]
    print nodelist
    X = model_w2v[nodelist]
    print X
    X2d = model_tsne.fit_transform(X)



    # print X2d
    clique_method = len(list(nx.k_clique_communities(G,2)))
    print clique_method
    k_means = KMeans(n_clusters=2, max_iter=100, precompute_distances=False)
    gmm = mixture.GMM(n_components=2,covariance_type='spherical',verbose=1)
    # print X
    gmm.fit(X)
    k_result = k_means.fit_predict(X)
    result = gmm.predict_proba(X)
    plt.scatter(X2d[:, 0], X2d[:, 1],c=15.0*k_result, marker='s')
    plt.show()
    print k_result
    print result
    print len(result)
   
    # for nd in node:
    #     print nd
    #     d = G[nd]
    #     # print d
    #     entry = {}
    #     entry['names'] = [key for key in d]
    #     # print entry['names']
    #     weights = [d[key]['weight'] for key in d]
    #     sumw = sum(weights)
    #     entry['weights'] = [i / sumw for i in weights]
    #     print entry['weights']

