### Overall Goal

Transform the `results_and_conclusions.tex` file and its associated plots from a draft state into a polished, publication-ready document suitable for a high-impact physics journal. The final section must be clear, self-contained, and visually appealing, ensuring a reader can fully understand our findings without referring to other parts of the paper.

### Context

The current `results_and_conclusions.tex` is the final part of a larger scientific report, with the preceding sections (introduction, methods, etc.) located in `content_report.tex`. The narrative and presentation of the results in `results_and_conclusions.tex` are not polished enough for publication. The plots, in particular, are poorly designed and need a complete overhaul to be clear, explanatory, and visually appealing.

### Key Files

*   `results_and_conclusions.tex`: **The Target File.** This is the file we will be incrementally editing.
*   `content_report.tex`: **The Context.** This file contains the problem setup, formulations, and methods. It must be read to understand the context of the results.
*   `plot_script_mapping.md`: **The Plot Roadmap.** This file maps each plot image to the Python script that generates it. This is our key to improving the plots.
*   `main.tex`: **The Root Document.** We need to consult this to understand the overall document structure, included packages (like `graphicx`, `booktabs`, `amsmath`, `subcaption`), and defined commands.
*   **Plot Generation Scripts (`.py`):** The Python scripts referenced in the mapping file, which we will need to edit.

---

### Plan of Attack (Sequential & Granular)

Follow these steps precisely. Do not move to the next step until the previous one is complete.

#### Phase 1: Analysis and Preparation (Read-Only)

1.  **Analyze Context:** Read `content_report.tex` thoroughly to understand the problem formulations, experimental setup, and methods. The results section must logically follow this context.
2.  **Analyze Current Results:** Read the existing `results_and_conclusions.tex`. Your goal is to understand the key findings, data, and structure that are already present. This is the raw material you will be polishing, not discarding.
3.  **Analyze Plot Map:** Read `plot_script_mapping.md` to create a list of all plots that need to be regenerated and their corresponding scripts.
4.  **Analyze LaTeX Preamble:** Read the preamble of `main.tex` to identify available packages we must use (e.g., `booktabs` for tables, `siunitx` for units, color packages).

#### Phase 2: Plot Regeneration (One Plot at a Time)

This is the most critical phase. We will regenerate each plot to be publication-quality.

5.  **Create a New Directory:** Create a new directory named `paper_plots/` to store the improved plot images. This avoids overwriting the old ones.
6.  **Iterate Through Plots:** For each plot identified in `plot_script_mapping.md`:
    a. **Read the script:** Open the corresponding Python script (e.g., `generate_method_comparison_plots.py`).
    b. **Modify the script for Quality:** Edit the plotting section of the script. Your goal is to make it look professional. Apply these changes:
        *   Increase the legibility: larger font sizes for titles, axis labels, and tick labels.
        *   Use a professional and colorblind-friendly color palette (e.g., `viridis`, `plasma`, or `cividis` from matplotlib).
        *   Ensure axis labels are descriptive and include units (e.g., "Energy (a.u.)").
        *   Add a clear, concise title if appropriate.
        *   Use thicker lines and larger markers for better visibility.
        *   Add a legend if multiple data series are present.
        *   Ensure the output is saved as a high-resolution format (e.g., PDF, or PNG with `dpi=300`).
    c. **Run the script:** Execute the modified script.
    d. **Save the Output:** Confirm that the new, high-quality plot image is saved in the `paper_plots/` directory with a descriptive name (e.g., `energy_vs_time_comparison.pdf`).

#### Phase 3: LaTeX Refactoring (Incremental Edits)

Now we will rewrite `results_and_conclusions.tex`, using the existing content as a basis but improving the structure, narrative, and visual elements (tables and plots).

7.  **Isolate Old Content:** In `results_and_conclusions.tex`, comment out the entire existing content by wrapping it in `\begin{comment} ... \end{comment}`. This preserves it as a reference.
8.  **Create New Sections:** Below the commented-out block, add `\section{Results}` and `\section{Discussion and Conclusions}`.
9.  **Rewrite the "Results" Section (Chunk by Chunk):** Go through the commented-out original text subsection by subsection (e.g., Hybrid Solvers, Pure QPU, Quantum Advantage). For each subsection:
    a. **Rewrite the Narrative:** Rewrite the explanatory text to be clearer, more concise, and in a professional, scientific tone. Ensure the story of the results is easy to follow.
    b. **Recreate Tables:** Rebuild the tables using the data from the original text. Use the `booktabs` package for a professional look (`\toprule`, `\midrule`, `\bottomrule`). Write new, clear, and self-contained captions.
    c. **Insert New Plots:** Insert the relevant, newly-generated plot(s) from the `paper_plots/` directory using a `figure` environment. Write a detailed, self-contained `\caption{}`. The caption must explain what the plot shows, define all symbols and lines, and state the key takeaway. Add a `\label{}` and reference it from the text with `\ref{}`.
10. **Rewrite the "Discussion and Conclusions" Section:** After rewriting all the results, tackle the final section. Synthesize the key findings you've just presented. Discuss their implications, compare them to the projections from the Full Proposal (as mentioned in the original text), and provide a strong, evidence-based concluding summary.


### Final Review Guidelines

*   **Tone:** Maintain a formal, objective, and scientific tone throughout.
*   **Clarity:** Ensure every sentence is clear and unambiguous.
*   **Self-Contained Captions:** A reader should be able to understand a figure or table just by reading its caption.
*   **Consistency:** Use consistent terminology and formatting.
*   **Check Compilation:** After each major step (e.g., adding a figure), assume you can compile the LaTeX document to check for errors.
