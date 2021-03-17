import re
import nltk
import heapq
import math

nltk.download('punkt')
nltk.download('stopwords')
language = "russian"  # 'english'
stopwords = nltk.corpus.stopwords.words(language)


def compute_tf(article_text):
	word_frequencies = {}
	for word in nltk.word_tokenize(article_text, language=language):
		if word not in stopwords:
			if word not in word_frequencies.keys():
				word_frequencies[word] = 1
			else:
				word_frequencies[word] += 1

	maximum_frequency = max(word_frequencies.values())

	for word in word_frequencies.keys():
		word_frequencies[word] = (word_frequencies[word] / maximum_frequency)

	return word_frequencies


def compute_idf(sentence_list):
	idf_measure = {}
	for sent_number, sent in enumerate(sentence_list):
		for word in nltk.word_tokenize(sent.lower(), language=language):
			if len(sent.split(' ')) < 30:
				if word not in idf_measure:
					idf_measure[word] = [sent_number]
				elif sent_number not in idf_measure[word]:
					idf_measure[word].append(sent_number)

	N = len(sentence_list)

	for key, value in idf_measure.items():
		idf_measure[key] = math.log(N / len(value))

	return idf_measure


def compute_tf_idf(article_text, idf_flag=True):
	sentence_list = nltk.sent_tokenize(article_text, language=language)
	tf_measure = compute_tf(article_text)
	idf_measure = dict.fromkeys(tf_measure.keys(), 1)
	if idf_flag:
		idf_measure = compute_idf(sentence_list)

	tf_idf_measure = {}
	for sent in sentence_list:
		for word in nltk.word_tokenize(sent.lower(), language=language):
			if word in tf_measure.keys():
				if len(sent.split(' ')) < 30:
					if sent not in tf_idf_measure.keys():
						tf_idf_measure[sent] = tf_measure[word] * idf_measure[word]
					else:
						tf_idf_measure[sent] += tf_measure[word] * idf_measure[word]

	return  tf_idf_measure


def predict_tfidf(text, result_sent_perc=10):
	article_text = re.sub(r'\[[0-9]*\]', ' ', text)
	article_text = article_text.replace('\r\n', ' <br> ')

	# formatted_article_text = re.sub('[^a-zA-Z]', ' ', article_text )
	# formatted_article_text = re.sub(r'\s+', ' ', formatted_article_text)

	sentence_list = nltk.sent_tokenize(article_text, language=language)
	sentence_score = compute_tf_idf(article_text)

	summary_sentences = heapq.nlargest(math.floor(len(sentence_list) * result_sent_perc / 100
											 ), sentence_score, key=sentence_score.get)
	# res = ' '.join(summary_sentences)
	res = ' '
	for sent in sentence_list:
		if sent in summary_sentences:
			res += " <strong>" + sent + "</strong>"
		else:
			res += " " + sent

	return res


def predict_tf(text, result_sent_perc=10):
	article_text = re.sub(r'\[[0-9]*\]', ' ', text)
	article_text = article_text.replace('\r\n', ' <br> ')

	sentence_list = nltk.sent_tokenize(article_text, language=language)
	sentence_score = compute_tf_idf(article_text,idf_flag=False)

	summary_sentences = heapq.nlargest(math.floor(len(sentence_list) * result_sent_perc / 100
											 ), sentence_score, key=sentence_score.get)
	res = ' '
	for sent in sentence_list:
		if sent in summary_sentences:
			res += " <strong>" + sent + "</strong>"
		else:
			res += " " + sent

	return res