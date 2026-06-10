run_api:
	uvicorn ceres_package.api.fast:app --reload

reinstall_package:
	@pip uninstall -y ceres_package || :
	@pip install -e .
