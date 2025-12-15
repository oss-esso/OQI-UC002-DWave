# Prompt for LLM Coding Agent: Gurobi Rotation Scenario Timeout Test

## Objective

Your task is to configure and run a Gurobi timeout test for a comprehensive set of crop rotation scenarios. This involves combining scenarios from multiple sources, updating a test script with a specific Gurobi configuration, and executing the test to verify timeout and performance behavior across different problem sizes.

You should use `mcp` tools for all file operations.

## Task List

1.  **Identify and Consolidate Scenarios:**
    *   Your first step is to identify all "rotation" scenarios available in the project.
    *   **Source 1:** Get all `rotation_*` scenarios from `src/scenarios.py`.
    *   **Source 2:** Get the family-only rotation scenarios defined in `@@todo/statistical_comparison_test.py`. These are for farm sizes `[5, 10, 15, 20, 25]` with `n_crops: 6` (which represent crop families).
    *   Combine these into a single, comprehensive list of scenarios to be tested. The final list should include scenarios with both 6 families and 27 foods.

2.  **Modify the Test Script (`@@todo/test_gurobi_timeout.py`):**
    *   Read the contents of the script `@@todo/test_gurobi_timeout.py`.
    *   **Update Gurobi Configuration:** Replace the existing `GUROBI_CONFIG` dictionary in the script with the following configuration:
        ```python
        GUROBI_CONFIG = {
            'timeout': 100,  # 100 seconds HARD LIMIT
            'mip_gap': 0.01,  # 1% - stop within 1% of optimum
            'mip_focus': 1,  # Find good feasible solutions quickly
            'improve_start_time': 30,  # Stop if no improvement for 30s
        }
        ```
    *   **Update Scenarios List:** Replace the hardcoded `SCENARIOS` list in the script with the comprehensive list you created in the first step. Ensure the scenario definitions are in the same format as the existing `SCENARIOS` list. You will need to add new entries for the scenarios that are not already present.

3.  **Execute the Test:**
    *   Run the modified `@@todo/test_gurobi_timeout.py` script.
    *   Ensure you capture all console output.

4.  **Report Results:**
    *   Summarize the results from the console output.
    *   List the scenarios that were tested.
    *   Report the final summary of timeout hits and `ImproveStartTime` stops.
    *   State the location of the generated JSON and CSV result files.