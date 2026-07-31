.PHONY: help build run run-query cluster clean

DATASET_URL = https://files.grouplens.org/datasets/movielens/ml-latest.zip

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

download-data: ## Download MovieLens dataset
	@mkdir -p data
	@echo "Downloading MovieLens dataset..."
	curl -L -o data/ml-latest.zip $(DATASET_URL)
	cd data && unzip -o ml-latest.zip && rm ml-latest.zip
	@echo "Dataset ready at data/ml-latest/"

build: ## Build Docker image
	docker compose build standalone

run: ## Run all queries (standalone)
	docker compose --profile standalone up --build

run-query: ## Run specific query (usage: make run-query Q=1)
	docker compose run --rm --build standalone --profile docker --query $(Q)

cluster: ## Run on Spark cluster (master + 2 workers)
	docker compose --profile cluster up --build

cluster-down: ## Stop Spark cluster
	docker compose --profile cluster down

clean: ## Remove output and Docker artifacts
	rm -rf output/*
	docker compose --profile standalone --profile cluster down --rmi local --volumes 2>/dev/null || true

sbt-compile: ## Compile locally via SBT (requires SBT installed)
	sbt compile

sbt-run: ## Run locally via SBT (usage: make sbt-run PROFILE=local)
	sbt "run --profile $(PROFILE) --query all"

sbt-assembly: ## Build fat JAR locally
	sbt assembly
