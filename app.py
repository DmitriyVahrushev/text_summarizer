import os

from flask import Flask, render_template, request, redirect, flash, jsonify
from algorithm_lda import predict_lda
from model_inference import predict_tf, predict_tfidf

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def upload_text():
	if request.method == 'POST':
		print(request.form)
		if 'input_text' not in request.form:
			return render_template('index.html', output_text = "Не было введено текста!")

		input_text = request.form['input_text']
		if not input_text:
			return render_template('index.html', output_text = "Не было введено текста!")
		output_text = predict_lda(input_text)
		return render_template('index.html', output_text = output_text)
	return render_template('index.html')


'''В POST-запросе должен быть json  с текстом. возвращает json c разметкой текста'''
@app.route('/api/get_predict', methods=['POST'])
def api_get_text():
	recieved_json = request.get_json()
	if recieved_json is None or 'input_text' not in recieved_json:
		return jsonify({'output_text':'No json sent'})
	if recieved_json['input_text'] is None:
		return jsonify({'output_text':'No text in input_text key'})
	input_text = recieved_json['input_text']
	output_text = predict_lda(input_text)
	recieved_json['output_text'] = output_text
	return jsonify(recieved_json)


'''Возвращает предсказания конкретной модели'''
@app.route('/api/<model_name>/get_predict', methods=['POST'])
def api_predict_with(model_name):
	recieved_json = request.get_json()
	if recieved_json is None or 'input_text' not in recieved_json:
		return jsonify({'output_text':'No json sent'})
	if recieved_json['input_text'] is None:
		return jsonify({'output_text':'No text in input_text key'})
	input_text = recieved_json['input_text']
	sentence_count = 10
	if 'sentence_count' in recieved_json:
		sentence_count = recieved_json['sentence_count']
	output_text = ''
	if model_name == 'tfidf':
		output_text = predict_tfidf(input_text, sentence_count)
	elif model_name == 'lda':
		output_text = predict_lda(input_text, sentence_count)
	elif model_name =='tf':
		output_text = predict_tf(input_text, sentence_count)
	else:
		output_text = 'No such model!!!'
	recieved_json['output_text'] = output_text
	return jsonify(recieved_json)


if __name__ == '__main__':
    app.run(debug=False, port=int(os.environ.get('PORT', 5000)))