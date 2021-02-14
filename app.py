import os

from flask import Flask, render_template, request, redirect, flash
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


if __name__ == '__main__':
    app.run(debug=True, port=int(os.environ.get('PORT', 5000)))