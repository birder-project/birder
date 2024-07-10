# Results Tool

The `results` tool is a powerful utility for analyzing and visualizing classification results in the Birder project. It allows researchers to process result files, compare multiple experiments and generate various metrics and visualizations.

## Usage

```sh
python -m birder.tools results [OPTIONS] RESULT_FILES...
```

Certainly! I'll add that information to the Usage section. Here's the updated version:

To see all available options and get detailed help, run:

```sh
python -m birder.tools results --help
```

This will display a comprehensive list of all options and their descriptions, ensuring you have access to the full functionality of the `results` tool.

## Description

The `results` tool reads and processes result files generated from classification experiments by the predict script. It can handle multiple result files simultaneously, allowing for easy comparison between different models or experimental setups.

Key features include:

* Printing detailed result tables
* Generating confusion matrices
* Plotting ROC curves
* Creating precision-recall curves
* Visualizing probability histograms
* Listing misclassifications and samples outside the top-k predictions

## Notes

* The results files contain raw probabilities, allowing for flexible post-hoc analysis and metric calculation
* The tool supports shell-style wildcards for specifying multiple result files or classes
* When comparing multiple results, the tool will generate a summary table for quick overview
* For visualization options (like ROC curves, precision-recall curves, and confusion matrices), you can specify which classes to include, allowing for focused analysis on specific species or groups
