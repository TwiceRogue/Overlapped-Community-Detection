import networkx as nx
import numpy as np
import numpy.random as npr
import gensim.models.word2vec as w2v
def alias_setup(probs):
    K = len(probs)
    q = np.zeros(K)
    J = np.zeros(K,dtype=np.int)
    smaller = []
    larger  = []
    for kk, prob in enumerate(probs):
        q[kk] = K * prob
        if q[kk] <1.0 :
            smaller.append(kk)
        else:
            larger.append(kk)
    while len(smaller)>0 and len(larger)>0:
        small = smaller.pop()
        large = larger.pop()

        J[small] =large
        q[large] = q[large] - (1.0 - q[small])

        if q[large]<1.0:
            smaller.append(large)
        else:
            larger.append(large)
    return J,q


def alias_draw(J, q):
    """
    This function is to help draw random samples from discrete distribution with specific weights,
    the code were adapted from the following source:
    https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/

    arguments:
    J, q: generated from alias_setup(prob)
    return:
    a random number ranging from 0 to len(prob)
    """
    K = len(J)
    # Draw from the overall uniform mixture.
    kk = int(np.floor(npr.rand() * K))
    # Draw from the binary mixture, either keeping the
    # small one, or choosing the associated larger one.
    if npr.rand() < q[kk]:
        return kk
    else:
        return J[kk]

def build_node_alias(G):
    nodes = G.nodes()
    nodes_rw = {}
    for nd in nodes:
        d = G[nd]  #access edges
        entry = {}
        entry['names'] = [key for key in d]                 # calculate the neighbor weights
        weights = [d[key]['weight'] for key in d]
        sumw = sum(weights)
        entry['weights'] = [i/sumw for i in weights]
        J,q = alias_setup(entry['weights'])
        entry['J'] = J
        entry['q'] = q
        nodes_rw[nd] = entry
    return nodes_rw

def create_rand_walks(S, num_paths, length_path, filename):
    # S = from  the bulid_node_alias
    # filename =  where to write the results
    # using exp(rating) as egde weight
    fwrite = open(filename,'w')
    nodes = S.keys()
    for nd in nodes:
        for i in range(num_paths):
            walk = [nd]
            for j in range(length_path):
                cur = walk[-1]
                next_nds = S[cur]['names']
                if len(next_nds) <1:
                    break
                else:
                    J = S[cur]['J']
                    q = S[cur]['q']
                    rd = alias_draw(J,q)
                    nextnd = next_nds[rd]
                    walk.append(nextnd)
            walk = [str(x) for x in walk]
            fwrite.write(" ".join(walk)+ "\n")
    fwrite.close()
    return 1

