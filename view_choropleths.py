"""
Quick viewer script to open all generated choropleth maps.
"""

import os
import webbrowser
import glob
from pathlib import Path

def main():
    output_dir = "choropleth_outputs"
    html_files = sorted(glob.glob(f"{output_dir}/*.html"))
    
    if not html_files:
        print("No HTML files found in choropleth_outputs/")
        return
    
    print("="*80)
    print("CHOROPLETH MAP VIEWER")
    print("="*80)
    print(f"\nFound {len(html_files)} maps:")
    print()
    
    for i, file in enumerate(html_files, 1):
        name = os.path.basename(file)
        size = os.path.getsize(file)
        print(f"{i:2d}. {name} ({size:,} bytes)")
    
    print("\n" + "="*80)
    print("Options:")
    print("  - Enter a number to open that map")
    print("  - Enter 'all' to open all maps")
    print("  - Enter 'farm' to open all farm maps")
    print("  - Enter 'patch' to open all patch maps")
    print("  - Enter 'q' to quit")
    print("="*80)
    
    while True:
        choice = input("\nYour choice: ").strip().lower()
        
        if choice == 'q':
            break
        elif choice == 'all':
            for file in html_files:
                print(f"Opening: {os.path.basename(file)}")
                webbrowser.open(f"file://{os.path.abspath(file)}")
            break
        elif choice == 'farm':
            farm_files = [f for f in html_files if 'Farm' in f]
            for file in farm_files:
                print(f"Opening: {os.path.basename(file)}")
                webbrowser.open(f"file://{os.path.abspath(file)}")
            break
        elif choice == 'patch':
            patch_files = [f for f in html_files if 'Patch' in f]
            for file in patch_files:
                print(f"Opening: {os.path.basename(file)}")
                webbrowser.open(f"file://{os.path.abspath(file)}")
            break
        elif choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(html_files):
                file = html_files[idx]
                print(f"Opening: {os.path.basename(file)}")
                webbrowser.open(f"file://{os.path.abspath(file)}")
            else:
                print("Invalid number!")
        else:
            print("Invalid choice!")

if __name__ == "__main__":
    main()
