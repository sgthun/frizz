# filepath: /home/sgthun/frizz/bus/Makefile
.PHONY: build run test clean

build:
	docker build -t image-scraper .

run:
	docker run --rm image-scraper

pyshell:
	docker run -it --rm image-scraper python -i scraper.py

test:
	docker run --rm image-scraper python -m unittest discover -s tests

clean:
	docker rmi image-scraper || true