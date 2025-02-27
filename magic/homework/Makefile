PHONY: build
build:
	docker build -t llagent .


.PHONY: run
run: build
	docker run -it --network=host llagent
