import os
from pathlib import Path

def print_tree(path, prefix="", ignore_dirs=None, max_depth=5, current_depth=0):
    if ignore_dirs is None:
        ignore_dirs = {'.git', '__pycache__', 'venv', '.idea', '.vscode', '__MACOSX'}
    
    if current_depth > max_depth:
        return
    
    path = Path(path)
    if not path.exists():
        return
    
    items = sorted(path.iterdir(), key=lambda x: (not x.is_dir(), x.name))
    
    for i, item in enumerate(items):
        is_last = i == len(items) - 1
        
        if item.is_dir() and item.name in ignore_dirs:
            continue
            
        if item.is_dir():
            print(f"{prefix}{'└── ' if is_last else '├── '}{item.name}/")
            new_prefix = prefix + ('    ' if is_last else '│   ')
            print_tree(item, new_prefix, ignore_dirs, max_depth, current_depth + 1)
        else:
            # Show file size
            size = item.stat().st_size
            if size < 1024:
                size_str = f"{size}B"
            elif size < 1024*1024:
                size_str = f"{size/1024:.1f}KB"
            else:
                size_str = f"{size/(1024*1024):.1f}MB"
            
            print(f"{prefix}{'└── ' if is_last else '├── '}{item.name} ({size_str})")

print("="*60)
print("YASEN-ALPHA PROJECT STRUCTURE")
print("="*60)
print()
print_tree(".", max_depth=5)
