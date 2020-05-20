# simple Makefile to simplify repetitive build env management on posix machines

PYTHON ?= python

.PHONY: clean
clean:
	$(PYTHON) setup.py clean
	rm -rf dist
	rm -rf build
	rm -rf .pytest_cache
	rm -rf snape.egg-info
	rm -rf .coverage.*

.PHONY: install
install:
	$(PYTHON) setup.py install

.PHONY: sdist
sdist:
	$(PYTHON) setup.py sdist

.PHONY: test-dependencies
test-dependencies:
	$(PYTHON) -m pip install coverage pytest pytest-cov flake8

.PHONY: test-unit
test-unit:
	$(PYTHON) -m pytest -v --durations=4 --cov-config .coveragerc --cov snape

# TODO: add linting
.PHONY: test
test: test-unit
