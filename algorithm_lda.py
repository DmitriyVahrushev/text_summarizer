import re
import nltk
import gensim
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


def comprute_lda(tokens):
    bigram_model = gensim.models.Phrases(tokens)
    trigram_model = gensim.models.Phrases(bigram_model[tokens], min_count=1)
    tokens = list(trigram_model[bigram_model[tokens]])
    # print(tokens)

    dictionary_LDA = gensim.corpora.Dictionary(tokens)
    # TODO: need found optimal parameter no_below
    dictionary_LDA.filter_extremes(no_below=2)
    corpus = [dictionary_LDA.doc2bow(token) for token in tokens]

    np.random.seed(123456)
    # TODO: maybe could change num_topics if it increase model`s
    num_topics = 20
    lda_model = gensim.models.LdaModel(
        corpus, num_topics=num_topics,
        id2word=dictionary_LDA,
        passes=10, alpha=[0.01] * num_topics,
        eta=[0.01] * len(dictionary_LDA.keys())
    )

    for i, topic in lda_model.show_topics(formatted=True, num_topics=num_topics):   # num_words=20
        print(str(i) + ": " + topic)
        print()
    # TODO: need finish lda algorithm

    top_topics = {}




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
    # print(tokens_sentences_lemmatized)
    comprute_lda(tokens_sentences_lemmatized)








text = """The Chrysler Building, the famous art deco New York skyscraper, will be sold for a small fraction of its previous sales price.
The deal, first reported by The Real Deal, was for $150 million, according to a source familiar with the deal.
Mubadala, an Abu Dhabi investment fund, purchased 90% of the building for $800 million in 2008.
Real estate firm Tishman Speyer had owned the other 10%.
The buyer is RFR Holding, a New York real estate company.
Officials with Tishman and RFR did not immediately respond to a request for comments.
It's unclear when the deal will close.
The building sold fairly quickly after being publicly placed on the market only two months ago.
The sale was handled by CBRE Group.
The incentive to sell the building at such a huge loss was due to the soaring rent the owners pay to Cooper Union, a New York college, for the land under the building.
The rent is rising from $7.75 million last year to $32.5 million this year to $41 million in 2028.
Meantime, rents in the building itself are not rising nearly that fast.
While the building is an iconic landmark in the New York skyline, it is competing against newer office towers with large floor-to-ceiling windows and all the modern amenities.
Still the building is among the best known in the city, even to people who have never been to New York.
It is famous for its triangle-shaped, vaulted windows worked into the stylized crown, along with its distinctive eagle gargoyles near the top.
It has been featured prominently in many films, including Men in Black 3, Spider-Man, Armageddon, Two Weeks Notice and Independence Day.
The previous sale took place just before the 2008 financial meltdown led to a plunge in real estate prices.
Still there have been a number of high profile skyscrapers purchased for top dollar in recent years, including the Waldorf Astoria hotel, which Chinese firm Anbang Insurance purchased in 2016 for nearly $2 billion, and the Willis Tower in Chicago, which was formerly known as Sears Tower, once the world's tallest.
Blackstone Group (BX) bought it for $1.3 billion 2015.
The Chrysler Building was the headquarters of the American automaker until 1953, but it was named for and owned by Chrysler chief Walter Chrysler, not the company itself.
Walter Chrysler had set out to build the tallest building in the world, a competition at that time with another Manhattan skyscraper under construction at 40 Wall Street at the south end of Manhattan. He kept secret the plans for the spire that would grace the top of the building, building it inside the structure and out of view of the public until 40 Wall Street was complete.
Once the competitor could rise no higher, the spire of the Chrysler building was raised into view, giving it the title."""

predict(text)