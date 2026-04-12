.PHONY: py1 py2 py3 py4 py5 py6 all

PYTHON := $(shell which python3 2>/dev/null || which python 2>/dev/null)
PYTHON := $(PYTHON) -X utf8

py1:
	$(PYTHON) python/drill_01_cloud_storage.py

py2:
	$(PYTHON) python/drill_02_file_chunker.py

py3:
	$(PYTHON) python/drill_03_hit_counter.py

py4:
	$(PYTHON) python/drill_04_permission_manager.py

py5:
	$(PYTHON) python/drill_05_web_crawler.py

py6:
	$(PYTHON) python/drill_06_search_index.py

all: py1 py2 py3 py4 py5 py6
