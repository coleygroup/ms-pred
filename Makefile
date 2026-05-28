.PHONY: test test-cpu test-gpu

test: test-cpu

test-cpu:
	pytest -m "not gpu" -ra

test-gpu:
	pytest -m gpu -ra
