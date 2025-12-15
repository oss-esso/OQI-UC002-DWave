#!/usr/bin/env python3
"""
Filter duplicate file entries in ANALYSIS_REPORT.md, keeping only the most recent version.
Files are considered duplicates if they have the same base name with different timestamps.
"""

import re
from pathlib import Path
from collections import defaultdict
from datetime import datetime

def extract_timestamp(filename):
    """Extract timestamp from filename (format: YYYYMMDD_HHMMSS)."""
    match = re.search(r'_(\d{8})_(\d{6})', filename)
    if match:
        date_str = match.group(1) + match.group(2)
        try:
            return datetime.strptime(date_str, '%Y%m%d%H%M%S')
        except:
            pass
    
    # Try just date (YYYYMMDD)
    match = re.search(r'_(\d{8})', filename)
    if match:
        try:
            return datetime.strptime(match.group(1), '%Y%m%d')
        except:
            pass
    
    return None

def get_base_name(filepath):
    """Get base name without timestamp."""
    # Remove timestamp patterns
    base = re.sub(r'_\d{8}_\d{6}', '', filepath)
    base = re.sub(r'_\d{8}', '', base)
    base = re.sub(r'_\d{10}', '', base)  # Unix timestamp
    return base

def parse_report(report_path):
    """Parse the report and extract file entries."""
    with open(report_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    # Split by file analysis sections
    sections = re.split(r'\n---\n', content)
    
    file_entries = []
    header = sections[0] if sections else ""
    
    for section in sections[1:]:
        match = re.search(r'## File Analysis: `([^`]+)`', section)
        if match:
            filepath = match.group(1)
            file_entries.append({
                'filepath': filepath,
                'section': '---\n' + section
            })
    
    return header, file_entries

def filter_duplicates(file_entries):
    """Filter file entries to keep only most recent versions."""
    # Group by base name
    groups = defaultdict(list)
    
    for entry in file_entries:
        filepath = entry['filepath']
        base_name = get_base_name(filepath)
        timestamp = extract_timestamp(filepath)
        
        groups[base_name].append({
            'entry': entry,
            'timestamp': timestamp,
            'filepath': filepath
        })
    
    # Keep only most recent from each group
    filtered = []
    duplicates_removed = []
    
    for base_name, items in groups.items():
        if len(items) == 1:
            # No duplicates
            filtered.append(items[0]['entry'])
        else:
            # Sort by timestamp (None timestamps go last)
            items_with_ts = [item for item in items if item['timestamp'] is not None]
            items_without_ts = [item for item in items if item['timestamp'] is None]
            
            if items_with_ts:
                items_with_ts.sort(key=lambda x: x['timestamp'], reverse=True)
                most_recent = items_with_ts[0]
                filtered.append(most_recent['entry'])
                
                # Track removed duplicates
                for item in items_with_ts[1:] + items_without_ts:
                    duplicates_removed.append(item['filepath'])
            else:
                # No timestamps, keep first one
                filtered.append(items[0]['entry'])
                for item in items[1:]:
                    duplicates_removed.append(item['filepath'])
    
    return filtered, duplicates_removed

def write_filtered_report(output_path, header, filtered_entries):
    """Write filtered report."""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(header)
        for entry in filtered_entries:
            f.write(entry['section'])

def main():
    report_path = Path('/Users/edoardospigarolo/Documents/OQI-UC002-DWave/@todo/ANALYSIS_REPORT.md')
    output_path = report_path  # Overwrite the original file
    backup_path = report_path.with_suffix('.md.backup')
    
    # Create backup
    print(f"Creating backup at {backup_path}...")
    import shutil
    shutil.copy(report_path, backup_path)
    
    print(f"Reading {report_path}...")
    header, file_entries = parse_report(report_path)
    print(f"Found {len(file_entries)} file entries")
    
    print("\nFiltering duplicates...")
    filtered_entries, duplicates_removed = filter_duplicates(file_entries)
    
    print(f"\nKept {len(filtered_entries)} unique files")
    print(f"Removed {len(duplicates_removed)} duplicate files")
    
    if duplicates_removed:
        print("\nRemoved files (showing first 30):")
        for filepath in sorted(duplicates_removed)[:30]:
            print(f"  - {filepath}")
        if len(duplicates_removed) > 30:
            print(f"  ... and {len(duplicates_removed) - 30} more")
    
    print(f"\nWriting filtered report back to {output_path}...")
    write_filtered_report(output_path, header, filtered_entries)
    
    print("\n✓ Done!")
    print(f"Original: {len(file_entries)} files")
    print(f"Filtered: {len(filtered_entries)} files")
    print(f"Reduction: {len(file_entries) - len(filtered_entries)} files ({100 * (len(file_entries) - len(filtered_entries)) / len(file_entries):.1f}%)")
    print(f"\nBackup saved at: {backup_path}")

if __name__ == '__main__':
    main()
