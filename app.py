import os

from flask import Flask, render_template, request, redirect, flash, jsonify
from model_inference import predict

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
		output_text = predict(input_text)
		return render_template('index.html', output_text = output_text)
	return render_template('index.html')


'''В POST-запросе должен быть json  с текстом. возвращает json c разметкой текста'''
@app.route('/api/get_predict', methods=['POST'])
def api_get_text():
	# взять текст
	recieved_json = request.get_json()
	if recieved_json is None or 'input_text' not in recieved_json:
		return jsonify({'output_text':'No json sent'})
	if recieved_json['input_text'] is None:
		return jsonify({'output_text':'No text in input_text key'})
	input_text = recieved_json['input_text']
	output_text = predict(input_text)
	recieved_json['output_text'] = output_text
	return jsonify(recieved_json)


if __name__ == '__main__':
    app.run(debug=True, port=int(os.environ.get('PORT', 5000)))