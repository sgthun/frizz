.PHONY: build run test clean

build:
	docker build -t school-trainer .

run:
	cp -r ../bus/img data
	for i in `ls data`; do cp data/$$i/*/*.jpg data/$$i; done

	docker run --rm -v `pwd`/data:/app/data school-trainer python src/main.py

test:
	docker run --rm school-trainer python -m unittest discover -s tests

clean:
	docker rmi school-trainer || true