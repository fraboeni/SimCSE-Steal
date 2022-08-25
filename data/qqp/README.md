# How to load files for QQP

Since the interface for QQP on huggingface is broken, we need to cheat a bit:

1. run $python download.py (which will throw an error but still download the correct files)
2. parse them with $python export_csv.py (which will use pandas to get the correct data out)