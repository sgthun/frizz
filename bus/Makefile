.PHONY: build run test pyshell clean

build:
	docker build -t image-scraper .

run:
	docker run --rm -v `pwd`/img:/app/img image-scraper

search:
	docker run --rm -v `pwd`/img:/app/img image-scraper python src/main.py search.txt

pyshell:
	docker run -it --rm -v `pwd`/img:/app/img image-scraper python -i scraper.py

test:
	docker run --rm image-scraper python -m unittest discover -s tests

clean:
	docker rmi image-scraper || true