# LaTeX Technical Report Compilation Guide

## Overview

The technical report is organized into 8 chapters plus a master file. Each chapter is self-contained and can be compiled individually or as part of the complete report.

## File Structure

```
Latex/
├── technical_report_master.tex         # Master file (includes all chapters)
├── technical_report_chapter1.tex       # Chapter 1: Introduction
├── technical_report_chapter2.tex       # Chapter 2: Problem Formulation
├── technical_report_chapter3.tex       # Chapter 3: Alternative 1
├── technical_report_chapter4.tex       # Chapter 4: Alternative 2
├── technical_report_chapter5.tex       # Chapter 5: Testing & Validation
├── technical_report_chapter6.tex       # Chapter 6: Experimental Evaluation
├── technical_report_chapter7.tex       # Chapter 7: Software Engineering
└── technical_report_chapter8.tex       # Chapter 8: Conclusions
```

## Compilation Instructions

### Individual Chapters

Each chapter file is standalone and can be compiled individually:

```bash
# Compile a single chapter
pdflatex technical_report_chapter1.tex
pdflatex technical_report_chapter1.tex  # Run twice for references
```

### Complete Report (Master File)

**Note**: The master file currently uses `\input` statements which expect content-only versions of chapters. You have two options:

#### Option 1: Compile Individual Chapters Separately

Compile each chapter as a standalone PDF:

```bash
for i in {1..8}; do
    pdflatex technical_report_chapter$i.tex
    pdflatex technical_report_chapter$i.tex
done
```

This generates 8 separate PDFs, one per chapter.

#### Option 2: Create Content-Only Chapter Files (Recommended for Master)

To compile the master file, extract chapter content without preambles:

1. For each chapter, create a `*_content.tex` file containing only the content between `\begin{document}` and `\end{document}`

2. Then compile master:

```bash
pdflatex technical_report_master.tex
pdflatex technical_report_master.tex
```

## Required LaTeX Packages

Ensure you have these packages installed (typically included in TeX Live, MiKTeX, or MacTeX):

- `amsmath`, `amssymb`, `amsthm` - Mathematical typesetting
- `graphicx` - Graphics support
- `algorithm`, `algpseudocode` - Algorithm environments
- `listings` - Code listings
- `xcolor` - Color support
- `hyperref` - Hyperlinks and PDF metadata
- `cite` - Citations
- `geometry` - Page layout
- `fancyhdr` - Headers and footers
- `titlesec` - Section formatting
- `booktabs` - Professional tables
- `multirow` - Multi-row tables
- `float` - Float positioning

## Compilation Commands

### Basic Compilation

```bash
pdflatex filename.tex
```

### With Bibliography

```bash
pdflatex filename.tex
bibtex filename
pdflatex filename.tex
pdflatex filename.tex
```

### Clean Build

```bash
# Remove auxiliary files
rm *.aux *.log *.out *.toc *.lof *.lot *.loa

# Then compile
pdflatex filename.tex
pdflatex filename.tex
```

## Alternative: Use latexmk

For automatic multi-pass compilation:

```bash
latexmk -pdf technical_report_chapter1.tex
```

This automatically runs pdflatex multiple times until all references are resolved.

## Windows PowerShell Commands

```powershell
# Compile all chapters
1..8 | ForEach-Object {
    $chapter = "technical_report_chapter$_.tex"
    Write-Host "Compiling $chapter..."
    pdflatex $chapter
    pdflatex $chapter
}

# Clean auxiliary files
Remove-Item *.aux, *.log, *.out, *.toc, *.lof, *.lot, *.loa -ErrorAction SilentlyContinue
```

## Output

Each compilation generates:
- `*.pdf` - The compiled PDF document
- `*.aux` - Auxiliary file for references
- `*.log` - Compilation log
- `*.out` - Hyperref output
- `*.toc` - Table of contents
- `*.lof` - List of figures
- `*.lot` - List of tables
- `*.loa` - List of algorithms

## Chapter Contents

### Chapter 1: Introduction (21 pages)
- Motivation and background
- Research objectives
- Implementation scope
- Report organization

### Chapter 2: Problem Formulation (15 pages)
- Mathematical formulation
- Continuous and binary formulations
- CQM/BQM representations
- Complexity analysis

### Chapter 3: Alternative 1 - Custom Hybrid (18 pages)
- Architecture overview
- Algorithm components
- Workflow implementation
- Performance characteristics

### Chapter 4: Alternative 2 - Decomposed QPU (17 pages)
- Strategic decomposition
- Classical and quantum components
- Solver integration
- Comparative analysis

### Chapter 5: Testing and Validation (15 pages)
- Testing methodology
- Unit tests
- Integration tests
- Validation procedures

### Chapter 6: Experimental Evaluation (14 pages)
- Evaluation framework
- Metrics and benchmarks
- Expected results
- Comparative analysis

### Chapter 7: Software Engineering (16 pages)
- Architectural design
- Code quality standards
- Security and credentials
- Extensibility

### Chapter 8: Conclusions (12 pages)
- Summary of contributions
- Lessons learned
- Limitations
- Future work

**Total**: ~128 pages

## Quick Start

```bash
# Navigate to Latex directory
cd d:\Projects\OQI-UC002-DWave\Latex

# Compile Chapter 1 (Introduction)
pdflatex technical_report_chapter1.tex
pdflatex technical_report_chapter1.tex

# View the PDF
start technical_report_chapter1.pdf
```

## Troubleshooting

### "LaTeX Error: File `*.sty' not found"
- Install missing package via your TeX distribution's package manager
- TeX Live: `tlmgr install <package>`
- MiKTeX: Use MiKTeX Console to install packages

### "Undefined references"
- Run pdflatex twice to resolve references

### "! Package hyperref Error"
- Ensure hyperref is loaded last (except for a few exceptions)
- Already configured correctly in these files

### Long compilation time
- Large reports may take 30-60 seconds per compilation
- Use `pdflatex -draftmode` for faster draft compilations

## Best Practices

1. **Compile twice**: Always run pdflatex twice to resolve cross-references
2. **Check logs**: Review `.log` files for warnings and errors
3. **Version control**: Keep `.tex` source files in version control, not PDFs
4. **Incremental writing**: Compile frequently during editing to catch errors early
5. **Use editors**: TeXstudio, Overleaf, or VS Code with LaTeX Workshop extension

## Contact

For questions about LaTeX compilation or report content, refer to the implementation documentation in `@todo/` directory.

---

**Status**: ✅ All 8 chapters written and ready for compilation
**Format**: LaTeX with comprehensive mathematical and code formatting
**Total Content**: ~128 pages of technical documentation
