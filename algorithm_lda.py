import re
import nltk
import gensim
import heapq
import numpy as np

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('stopwords')
# TODO: change languages is not available
language = "english"
stopwords = nltk.corpus.stopwords.words(language)


def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return nltk.corpus.wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return nltk.corpus.wordnet.ADJ
    elif treebank_tag.startswith('N'):
        return nltk.corpus.wordnet.ADJ
    elif treebank_tag.startswith('R'):
        return nltk.corpus.wordnet.ADJ
    else:
        return ''


def comprute_lda(sentence_list, tokens):
    bigram_model = gensim.models.Phrases(tokens)
    trigram_model = gensim.models.Phrases(bigram_model[tokens], min_count=1)
    tokens = list(trigram_model[bigram_model[tokens]])

    dictionary_LDA = gensim.corpora.Dictionary(tokens)
    # TODO: need found optimal parameter no_below
    dictionary_LDA.filter_extremes(no_below=2)
    corpus = [dictionary_LDA.doc2bow(token) for token in tokens]

    np.random.seed(123456)
    # TODO: maybe could change num_topics if it increase model`s
    num_topics = 10
    lda_model = gensim.models.LdaModel(
        corpus, num_topics=num_topics,
        id2word=dictionary_LDA,
        passes=10, alpha=[0.01] * num_topics,
        eta=[0.01] * len(dictionary_LDA.keys())
    )

    # TODO: need finish lda algorithm

    topn = int(len(tokens) * 0.5)
    sentence_score = {}

    for topic in range(lda_model.num_topics):
        for word, probability in [_ for _ in lda_model.show_topic(topic, topn=topn)]:
            for sent in sentence_list:
                if word in sent:
                    if sent not in sentence_score:
                        sentence_score[sent] = probability
                    else:
                        sentence_score[sent] += probability

    return sentence_score


def predict(text):
    article_text = re.sub(r'\[[0-9]*\]', ' ', text)
    article_text = re.sub(r'\s+', ' ', article_text)
    # TODO: rename variables
    sentence_list = nltk.sent_tokenize(article_text, language=language)
    words_list = [nltk.word_tokenize(sent.lower(), language=language) for sent in sentence_list]

    list_tokens_pos = [nltk.pos_tag(word) for word in words_list]
    lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()
    tokens_sentences_lemmatized = [
        [
            lemmatizer.lemmatize(el[0], get_wordnet_pos(el[1]))
            if get_wordnet_pos(el[1]) != '' else el[0] for el in tokens_POS
        ]
        for tokens_POS in list_tokens_pos
    ]

    tokens_sentences_lemmatized = [
        [
            token.lower() for token in token_sentence if len(token)>1 and token.lower() not in stopwords
        ]
        for token_sentence in tokens_sentences_lemmatized
    ]
    sentence_score = comprute_lda(sentence_list, tokens_sentences_lemmatized)
    summary_sentences = heapq.nlargest(7, sentence_score, key=sentence_score.get)

    res = ' '
    for sent in sentence_list:
        if sent in summary_sentences:
            res += " <strong>" + sent + "</strong>"
        else:
            res += " " + sent

    return res
