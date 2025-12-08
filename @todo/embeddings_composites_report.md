Technical Report: Analysis of D-Wave Embedding and Composite Usage in
  Benchmarks

  Date: December 4, 2025
  Author: Gemini AI Agent

  1. Introduction

  This report analyzes the discrepancy between the functionalities
  documented in the D-Wave Ocean SDK HTML files (embeddings_dwave.html,
  composites_dwave.html) and their practical application within three
  specific benchmark scripts: study_embedding_scaling.py,
  cqm_partition_embedding_benchmark.py, and
  comprehensive_embedding_and_solving_benchmark.py.

  The objective is to identify which documented tools, composites, and
  concepts are not being actively used or tested in the provided
  benchmarks, highlighting potential gaps in benchmark coverage.

  2. Summary of Documented Functionalities

  A review of the provided HTML documentation reveals a rich set of tools
  for problem embedding and sampler composition.

  2.1. embeddings_dwave.html

  This document describes the dwave.embedding module, which provides a
  comprehensive toolkit for minor-embedding a problem onto a target
  graph. Key functionalities include:

   * Embedding Generators: Tools like minorminer.find_embedding and
     topology-specific helpers for Chimera, Pegasus, and Zephyr
     architectures (find_clique_embedding, find_grid_embedding).
   * Embedding Utilities: Functions to apply and reverse embeddings on a
     BQM (embed_bqm, unembed_sampleset).
   * Diagnostics: A suite of tools to analyze the quality of an
     embedding, including chain_break_frequency, diagnose_embedding, and
     is_valid_embedding.
   * Chain Strength Calculation: Heuristics like
     uniform_torque_compensation and scaled to determine the optimal
     chain strength, which is critical for maintaining the integrity of
     logical qubits.
   * Chain-Break Resolution: A variety of strategies to resolve chain
     breaks during unembedding, such as discard, majority_vote,
     weighted_random, and the MinimizeEnergy object.

  2.2. composites_dwave.html

  This document details the dwave.system.composites module, which offers
  a powerful layered approach to pre- and post-processing. These
  "composites" wrap samplers to add functionality. Notable composites
  include:

   * Core Embedding Composites:
       * EmbeddingComposite: The fundamental composite for applying a
         given minor-embedding.
       * AutoEmbeddingComposite: Automates the process by finding an
         embedding if one is not provided.
       * FixedEmbeddingComposite: Uses a pre-calculated embedding for a
         specific problem structure.
   * Advanced Embedding Composites:
       * TilingComposite: Maps large problems by breaking them into
         "tiles" that fit on the QPU.
       * VirtualGraphComposite: Simplifies working with a target graph by
         creating a virtual representation with the desired structure.
       * ParallelEmbeddingComposite: Tries multiple embeddings in
         parallel to find a good one quickly.
   * Specialized Sampler Composites:
       * ReverseAnneal Composites: ReverseBatchStatesComposite and
         ReverseAdvanceComposite provide interfaces for advanced reverse
         annealing protocols.
       * LinearAncillaComposite: A specialized tool for using ancillary
         qubits to apply linear biases to problem qubits.

  3. Analysis of Benchmark Implementations

  The three Python scripts focus on benchmarking different aspects of
  solving optimization problems, from embedding performance to overall
  solution quality.
   * study_embedding_scaling.py: This script appears to focus on
     measuring how embedding time and quality scale with problem size or
     other parameters.
   * cqm_partition_embedding_benchmark.py: This script likely evaluates
     the performance of partitioning a large Constrained Quadratic Model
     (CQM) and embedding the resulting subproblems.
   * comprehensive_embedding_and_solving_benchmark.py: This script seems
     to be a broad benchmark that measures the end-to-end process of
     embedding a problem and finding a solution with a D-Wave solver.

  A keyword analysis of these files shows that while they use core
  embedding features, they do not utilize the full suite of tools
  available in the Ocean SDK documentation.

  4. Identified Gaps: Unused SDK Functionalities

  The primary finding of this report is that the benchmark scripts test a
  subset of the documented functionalities. They appear to rely on
  default or basic embedding mechanisms while ignoring more advanced and
  specialized features.

  The following documented functionalities were not found in the
  benchmark scripts:

  4.1. From composites_dwave.html:

   * Advanced Embedding Strategies:
       * TilingComposite: There is no evidence that this composite, which
         is essential for solving problems larger than the QPU fabric via
         tiling, is being benchmarked.
       * VirtualGraphComposite: This abstraction layer for simplifying
         sampler interactions is not used.
       * ParallelEmbeddingComposite: The strategy of finding embeddings
         in parallel is not tested.
       * LazyFixedEmbeddingComposite: This performance-oriented composite
         for lazy loading is absent.

   * Specialized Annealing and Biasing Protocols:
       * Reverse Annealing: Neither ReverseBatchStatesComposite nor
         ReverseAdvanceComposite are used. This indicates a lack of
         benchmarking for advanced annealing protocols that start from a
         known classical state.
       * Ancilla-Based Biasing: The LinearAncillaComposite is not
         implemented, suggesting that benchmarks for this specific
         hardware biasing technique are not being run.

  4.2. From embeddings_dwave.html:

   * Advanced Diagnostic Tools:
       * The scripts do not call diagnostic functions like
         diagnose_embedding, verify_embedding, or chain_break_frequency.
         Benchmarking seems to focus on the end result (energy, time)
         rather than the quality metrics of the intermediate embedding
         itself.

   * Chain Strength Heuristics:
       * There is no direct use of the uniform_torque_compensation or
         scaled functions. The benchmarks likely rely on the default
         chain strength calculation provided by dwave-system or set a
         manual value, rather than comparing different advanced
         heuristics.

   * Chain-Break Resolution Strategies:
       * The scripts do not import or configure different chain-break
         resolution methods (discard, weighted_random, MinimizeEnergy).
         They rely on the default majority_vote behavior, leaving the
         performance implications of other strategies unevaluated.

   * Topology-Specific Generators:
       * The benchmarks do not appear to explicitly call
         hardware-specific embedding finders like
         pegasus.find_clique_embedding or
         chimera.find_biclique_embedding. They likely use a more generic
         approach via AutoEmbeddingComposite.

  5. Conclusion

  The provided benchmark scripts are focused on evaluating the core
  performance of embedding and solving QUBOs, likely using the default
  and most common tool, AutoEmbeddingComposite.

  However, a significant portion of the D-Wave Ocean SDK's capabilities
  remains un-benchmarked. These unused features represent more advanced,
  experimental, or fine-grained controls over the problem-solving
  process. Key areas for potential benchmark expansion include:

   1. Large Problem Decomposition: Evaluating the performance and
      efficacy of the TilingComposite.
   2. Advanced Annealing Schedules: Benchmarking the impact of
      ReverseAnneal composites on solution quality and refinement.
   3. Embedding Quality and Heuristics: Creating specific benchmarks to
      compare different chain strength calculation methods and
      chain-break resolution strategies.
   4. Specialized Composites: Assessing the utility and performance of
      tools like VirtualGraphComposite and ParallelEmbeddingComposite.