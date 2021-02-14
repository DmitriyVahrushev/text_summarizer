from summarizer import Summarizer

model = Summarizer()

def predict(text):
	result = model(text, ratio=0.2)
	#full = ''.join(result)
	orig_sentences = text.split(".") # use something smarter for this in the future
	res = ''
	for sentence in orig_sentences:
		if sentence in result:
			res += " <strong>" + sentence + "</strong>"
		else:
			res += " "+ sentence
	return res