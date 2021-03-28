from itertools import combinations
import nltk
import networkx as nx
import re
import math


def similarity(s1, s2):
    if not len(s1) or not len(s2):
        return 0.0
    return len(s1.intersection(s2))/(len(s1) + len(s2))


def compute_textrank(sentence_list):
    tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
    lmtzr = nltk.stem.snowball.RussianStemmer()
    words = [set(lmtzr.stem(word) for word in tokenizer.tokenize(sentence.lower()))
             for sentence in sentence_list]
    print(words)
    sentence_pairs = combinations(range(len(sentence_list)), 2)
    scores = [(i, j, similarity(words[i], words[j])) for i, j in sentence_pairs]
    scores = filter(lambda x: x[2], scores)

    g = nx.Graph()
    g.add_weighted_edges_from(scores)
    pr = nx.pagerank(g)

    return sorted(((i, pr[i], s) for i, s in enumerate(sentence_list) if i in pr),
                key=lambda x: pr[x[0]], reverse=True)


def predict_textrank(text, result_sent_perc= 50):
    article_text = re.sub(r'\[[0-9]*\]', ' ', text)
    article_text = article_text.replace('\r\n', ' <br> ')
    sentence_list = nltk.tokenize.sent_tokenize(article_text)
    sentence_score = compute_textrank(sentence_list)

    top_n = math.floor(len(sentence_list) * result_sent_perc / 100)
    summary_sentences = [sentence_score[i][2] for i in range(top_n)]

    res = ' '
    for sent in sentence_list:
        if sent in summary_sentences:
            res += " <strong>" + sent + "</strong>"
        else:
            res += " " + sent

    return res
