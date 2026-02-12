import sys

file_path = 'src/Dashboard.jsx'

with open(file_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

print(f"Total lines: {len(lines)}")

count = 0
for i, line in enumerate(lines):
    if "Manual Move" in line:
        count += 1
        print(f"Match {count} at line {i+1}: {line.strip()}")
        # print context
        print(f"  Context before: {lines[i-1].strip()}")
        print(f"  Context after:  {lines[i+1].strip()}")

if count == 0:
    print("No 'Manual Move' found in file!")
elif count == 1:
    print("Only 1 'Manual Move' found. No duplicates?")
else:
    print(f"Found {count} occurrences. Investigating duplicates...")
