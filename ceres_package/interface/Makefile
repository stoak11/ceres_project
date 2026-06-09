# Run from repo root:  make -f ceres_package/interface/Makefile <target>
ROOT := $(abspath $(dir $(lastword $(MAKEFILE_LIST)))/../..)
export PYTHONPATH := $(ROOT)
export PULL_FROM_GCP ?= false

.PHONY: help run smoke ping predict

help:
	@echo "Ceres predict API"
	@echo ""
	@echo "  make -f ceres_package/interface/Makefile run     Start API"
	@echo "  make -f ceres_package/interface/Makefile smoke    Offline test (no server)"
	@echo "  make -f ceres_package/interface/Makefile ping     curl /ping (server must be up)"
	@echo "  make -f ceres_package/interface/Makefile predict  curl example /predict"

run:
	$(ROOT)/ceres_package/interface/run.sh

smoke:
	cd $(ROOT) && python -m ceres_package.interface

ping:
	curl -s http://$${CERES_API_HOST:-127.0.0.1}:$${CERES_API_PORT:-8000}/ping | python -m json.tool

predict:
	curl -s "http://$${CERES_API_HOST:-127.0.0.1}:$${CERES_API_PORT:-8000}/predict?dept=51&year=2023" | python -m json.tool
