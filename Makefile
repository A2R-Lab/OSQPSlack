# Run inside an activated virtual environment (see README).
PYTHON ?= python

.PHONY: all install data figures

all: data figures

install:
	SKBUILD_CMAKE_DEFINE="OSQP_SOURCE_DIR=$(CURDIR)/external/osqp" \
		$(PYTHON) -m pip install -r requirements.txt

data:
	$(PYTHON) experiments/scaling_run.py
	$(PYTHON) experiments/energy_hub_run.py

figures:
	$(PYTHON) experiments/slack_projection.py
	$(PYTHON) experiments/scaling_plot.py
	$(PYTHON) experiments/energy_hub_plot.py
