# NLUTestFramework - A framework to benchmark and compare NLU frameworks.

This framework offers a simple interface to benchmark and compare the intent classification performance of various NLU frameworks. The performance is measured across a configurable number of iterations with the result being the mean and variance of the achieved F1 scores.
Each framework is benchmarked on one or more configurable data sets, which are randomly split into training and validation data on each iteration. The frameworks, data sets and the benchmarking behaviour are fully configurable in a single configuration file.

## Getting Started

Information about the installation and configuration of this benchmarking framework is located [in the docs](https://emundo.github.io/nlutestframework_doc/).

## Data

The exemplar data is taken from [Braun et al. 2017](https://github.com/sebischair/NLU-Evaluation-Corpora) and released under the CC BY-SA 3.0 license.
