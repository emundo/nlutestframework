define SETUP_ENV
python3 -m virtualenv env/
source env/bin/activate
python -m pip install --upgrade pip
python -m pip install --upgrade setuptools wheel
python -m pip install --upgrade pytest
deactivate
endef
export SETUP_ENV

define RUN_TESTS
source env/bin/activate
python setup.py install
python -m pytest
deactivate
endef
export RUN_TESTS

.PHONY: all
all: test

.PHONY: todo
todo:
	grep -rniI . -e todo --exclude-dir env --exclude-dir build --exclude-dir *.egg-info --exclude-dir _build --exclude Makefile

env/:
	bash -c "$$SETUP_ENV"

.PHONY: test
test: env/
	bash -c "$$RUN_TESTS"
