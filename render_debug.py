import os
print("=== DEBUG INFO ===")
print("Current working directory:", os.getcwd())
print("Files and folders:", os.listdir('.'))
for root, dirs, files in os.walk('.'):
    print(root, dirs, files)
print("===================")
