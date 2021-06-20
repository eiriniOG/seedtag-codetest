install_windows:
	python -m venv venv
	source venv/Scripts/activate
	pip install -r requirements.txt
	@echo "🚀 INSTALL WDEPS ok"

install_unix:
	python -m venv venv
	source venv/bin/activate
	pip install -r requirements.txt
	@echo "🚀 INSTALL UDEPS ok"

run:
	python src/model.py