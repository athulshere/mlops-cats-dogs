.PHONY: verify install prepare train test serve docker-build docker-run smoke monitor k8s-deploy clean

verify:
	python scripts/verify_setup.py

install:
	pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements-dev.txt

prepare:
	python -m src.data_prep

train:
	python -m src.train

test:
	pytest -v

serve:
	uvicorn app.main:app --reload --port 8000

docker-build:
	docker build -t cats-dogs-api:local .

docker-run:
	docker compose up -d

smoke:
	python scripts/smoke_test.py --url http://localhost:8000

monitor:
	python scripts/monitor_batch.py --url http://localhost:8000 --per-class 25

k8s-deploy:
	kubectl apply -f k8s/deployment.yaml -f k8s/service.yaml
	kubectl rollout status deployment/cats-dogs-api

clean:
	docker compose down
	rm -rf __pycache__ .pytest_cache
