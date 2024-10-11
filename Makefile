install:
	python3 -m venv venv
	. venv/bin/activate && pip install -r requirements.txt

run:
	. venv/bin/activate && export FLASK_APP=app.py && flask run --host=0.0.0.0 --port=3000

