.PHONY: py1 py2 py3 py4 py5 py6 ts1 ts2 ts3 ts4 ts5 ts6 all-py all-ts all

py1:
	python3 python/drill_01_cloud_storage.py

py2:
	python3 python/drill_02_file_chunker.py

py3:
	python3 python/drill_03_hit_counter.py

py4:
	python3 python/drill_04_permission_manager.py

py5:
	python3 python/drill_05_web_crawler.py

py6:
	python3 python/drill_06_search_index.py

ts1:
	npx tsx typescript/drill_01_cloud_storage.ts

ts2:
	npx tsx typescript/drill_02_file_chunker.ts

ts3:
	npx tsx typescript/drill_03_hit_counter.ts

ts4:
	npx tsx typescript/drill_04_permission_manager.ts

ts5:
	npx tsx typescript/drill_05_web_crawler.ts

ts6:
	npx tsx typescript/drill_06_search_index.ts

all-py: py1 py2 py3 py4 py5 py6
all-ts: ts1 ts2 ts3 ts4 ts5 ts6
all: all-py all-ts
