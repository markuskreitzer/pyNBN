# Neuron-by-Neuron Training and Modern LLM Applications

## Research Document

This document explores the potential applications of the Neuron-by-Neuron (NBN) training algorithm and cascading neural network concepts to modern Large Language Model (LLM) training. The NBN algorithm was developed by Dr. B.M. Wilamowski at Auburn University and offers several innovative approaches that have relevance to contemporary deep learning challenges.

---

## Table of Contents

1. [Overview of NBN Algorithm](#overview-of-nbn-algorithm)
2. [Key Concepts in pyNBN](#key-concepts-in-pynbn)
3. [Relevance to Modern LLM Training](#relevance-to-modern-llm-training)
4. [Research Connections](#research-connections)
5. [Current Research Directions](#current-research-directions)
6. [Challenges and Opportunities](#challenges-and-opportunities)
7. [Conclusion](#conclusion)
8. [References](#references)

---

## Overview of NBN Algorithm

The Neuron-by-Neuron (NBN) algorithm is an innovative neural network training approach with several distinctive features:

### Core Principles

1. **Forward-Only Computation**: Unlike traditional backpropagation, NBN can train networks using forward passes only, eliminating the need for backward error propagation through the network.

2. **Neuron-Level Training**: Rather than updating all weights simultaneously, NBN processes each neuron individually, computing gradients and Hessians on a per-neuron basis.

3. **Arbitrarily Connected Networks**: NBN can train networks with arbitrary connection patterns (skip connections, cross-layer links), not just standard layer-by-layer architectures.

4. **Direct Quasi-Hessian Computation**: The algorithm directly computes the quasi-Hessian matrix neuron-by-neuron, reducing memory requirements since the full Jacobian matrix doesn't need to be stored.

5. **Second-Order Optimization**: NBN uses Levenberg-Marquardt optimization, which leverages second-order curvature information for faster convergence.

### Algorithm Advantages

- **Memory Efficiency**: By not requiring storage of the full Jacobian, NBN can train larger networks with limited memory.
- **Flexible Architectures**: Support for fully connected cascade (FCC), multi-layer perceptron (MLP), and other network topologies.
- **Convergence Speed**: Second-order methods typically converge faster than first-order gradient descent.

---

## Key Concepts in pyNBN

The pyNBN implementation supports several network topologies:

### Supported Architectures

| Topology | Description | Example |
|----------|-------------|---------|
| **MLP** | Multi-Layer Perceptron | `[7, 3, 4, 2, 1]` - 7 inputs, hidden layers of 3, 4, 2 neurons, 1 output |
| **SLP** | Single-Layer Perceptron | `[7, 17, 1]` - 7 inputs, 17 hidden neurons, 1 output |
| **FCC** | Fully Connected Cascade | `[7, 1, 1, 1, 1, 1, 1]` - Each neuron connects to all previous neurons |
| **BMLP** | Bridged MLP | Similar to FCC with bridged connections |

### Training Components

1. **Hessian Computation** (`hessian.py`): Computes gradients and Hessian matrices neuron-by-neuron
2. **Forward Calculation** (`calc_fwd.py`): Computes network outputs through forward passes
3. **Error Calculation** (`calculate_error.py`): Computes Sum of Squared Errors (SSE)
4. **Activation Functions** (`actFunc.py`): Supports linear, unipolar sigmoid, bipolar (tanh), and Elliot functions

---

## Relevance to Modern LLM Training

### 1. Forward-Only Training Paradigms

The NBN concept of forward-only computation has modern parallels in recent research:

#### NoProp Algorithm (2025)
- **Paper**: "NoProp: Training Neural Networks without Full Back-propagation or Full Forward-propagation"
- **Key Finding**: Networks can be trained where each block (layer) learns independently using local targets, without global error propagation.
- **Relevance**: This directly echoes NBN's neuron-level training philosophy.
- **Results**: Demonstrated on MLPs, CNNs, and Transformers with up to 20% speedup on deeper networks.
- **Link**: [arXiv:2503.24322](https://arxiv.org/abs/2503.24322)

#### Mono-Forward Algorithm (2024)
- **Paper**: "Mono-Forward: Backpropagation-Free Algorithm for Efficient Neural Network Training Harnessing Local Errors"
- **Concept**: Inspired by Hinton's Forward-Forward framework, optimizes layers using exclusively local error information.
- **Link**: [HuggingFace Papers](https://huggingface.co/papers/2501.09238)

#### Forward-Forward Algorithm (Hinton, 2022)
- **Paper**: "The Forward-Forward Algorithm: Some Preliminary Investigations"
- **Concept**: Replaces backpropagation with two forward passes using positive/negative data.
- **Local Learning**: Each layer has its own objective, similar to NBN's per-neuron approach.
- **Link**: [cs.toronto.edu](https://www.cs.toronto.edu/~hinton/FFA13.pdf)

### 2. Second-Order Optimization for LLMs

NBN's use of Levenberg-Marquardt (a second-order method) has parallels in modern LLM optimization research:

#### Full Gauss-Newton for LLMs (2025)
- **Paper**: "The Potential of Second-Order Optimization for LLMs: A Study with Full Gauss-Newton"
- **Key Finding**: Full Gauss-Newton updates delivered 5.4x reduction in training steps compared to baselines.
- **Insight**: Layerwise GN preconditioning (similar to NBN's approach) nearly matched full GN performance.
- **Link**: [arXiv:2510.09378](https://arxiv.org/abs/2510.09378)

#### HiZOO: Hessian-Informed Zeroth-Order Optimizer (ICLR 2025)
- **Paper**: "Second-Order Fine-Tuning without Pain for LLMs"
- **Concept**: Uses diagonal Hessian information for memory-efficient LLM fine-tuning.
- **Link**: [OpenReview](https://openreview.net/forum?id=bEqI61iBue)

#### Modern Second-Order Optimizers
| Optimizer | Approach | Application |
|-----------|----------|-------------|
| SOAP | Layerwise preconditioning | LLM training |
| Shampoo | Matrix preconditioners | Large-scale models |
| Muon | Block-diagonal Hessian | Transformer training |

### 3. Arbitrarily Connected Networks

NBN's support for arbitrary network connections relates to modern architecture innovations:

#### Skip Connections in Transformers
- Residual connections in transformers are a form of arbitrary connectivity.
- Allow gradients to flow across many layers, stabilizing training of deep models.
- **Reference**: [IEEE: Why Is Everyone Training Very Deep Neural Network With Skip Connections?](https://ieeexplore.ieee.org/document/9669990)

#### Brain-Inspired Sparse Training (2025)
- **Paper**: "Brain network science modelling of sparse neural networks enables scalable training"
- **Key Finding**: Extremely sparse architecture (<1% connectivity) can match dense performance.
- **Relevance**: NBN's arbitrary connectivity allows similar flexibility.
- **Link**: [arXiv:2501.19107](https://arxiv.org/abs/2501.19107)

#### Mixture of Experts (MoE)
- MoE models use sparse, modular designs similar to cascading architectures.
- Only selected "experts" (sub-networks) activate for each input.
- **Reference**: [University of Washington: Mixture-of-Experts in the Era of LLMs](https://courses.cs.washington.edu/courses/cse599k/24au/content/MoE.pdf)

---

## Research Connections

### Conceptual Mapping: NBN → Modern LLM Techniques

| NBN Concept | Modern Equivalent | Status |
|------------|-------------------|--------|
| Forward-only computation | NoProp, Forward-Forward algorithms | Active research |
| Per-neuron gradient computation | Local learning rules | Emerging |
| Quasi-Hessian direct computation | Layerwise GN, diagonal Hessian | Proven effective |
| Arbitrary network connectivity | Skip connections, MoE, sparse networks | Standard practice |
| Second-order optimization (L-M) | SOAP, Shampoo, Muon, HiZOO | Active development |
| Layer-wise training | Greedy layer-wise, progressive growing | Revisited for efficiency |

### Why These Connections Matter

1. **Memory Efficiency**: Both NBN and modern forward-only methods address memory constraints in training deep networks.

2. **Parallelization**: Per-neuron/per-layer training enables greater parallelism, crucial for distributed LLM training.

3. **Biological Plausibility**: NBN's local computation aligns with brain-inspired learning research gaining traction for edge AI.

4. **Convergence Speed**: Second-order methods (like NBN's L-M) show 5-16x faster convergence in recent LLM studies.

---

## Current Research Directions

### 1. Backpropagation-Free Training for LLMs

A comprehensive survey (2024) reviews methods avoiding backpropagation in LLMs:
- **Paper**: "A Survey of Backpropagation-Free Training For LLMs"
- **Finding**: Forward-only methods substantially lower memory requirements but face scalability challenges.
- **Link**: [Beijing University Survey PDF](https://d197for5662m48.cloudfront.net/documents/publicationstatus/201426/preprint_pdf/883fb6f1ba80815ebf0e357734d27f40.pdf)

### 2. Layer-Wise Training Revival

#### Greedy Layer-Wise Training
- **Paper**: "Greedy Layerwise Learning Can Scale to ImageNet"
- **Finding**: Layer-wise training is competitive with end-to-end training at scale.
- **Link**: [PMLR](https://proceedings.mlr.press/v97/belilovsky19a/belilovsky19a.pdf)

#### GLNAS: Architecture Search
- **Paper**: "GLNAS: Greedy Layer-wise Network Architecture Search"
- **Concept**: Uses greedy layer-wise methods for efficient neural architecture search.
- **Link**: [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0031320324004813)

### 3. Progressive Growing Networks

- **Paper**: "GrowNN: Growing with Experience in Deep RL" (2025)
- **Concept**: Networks expand dynamically as training progresses.
- **Relevance**: Similar to NBN's cascading network concept.
- **Link**: [arXiv:2506.11706](https://arxiv.org/abs/2506.11706)

---

## Challenges and Opportunities

### Challenges for Scaling NBN Concepts to LLMs

1. **Computational Overhead**: Full Hessian computation is infeasible for billions of parameters.
   - **Mitigation**: Use diagonal or block-diagonal approximations (layerwise GN).

2. **Performance Gap**: Forward-only methods currently lag backpropagation for large-scale tasks.
   - **Ongoing Work**: NoProp and enhanced FF algorithms are closing this gap.

3. **Infrastructure**: Modern deep learning frameworks are optimized for backpropagation.
   - **Opportunity**: Custom implementations (like pyNBN) can explore alternatives.

### Opportunities

1. **Low-Power Training**: Forward-only and local learning methods suit edge devices and neuromorphic hardware.

2. **Scalable Second-Order Methods**: Layerwise and diagonal Hessian approaches retain most convergence benefits at reasonable cost.

3. **Hybrid Approaches**: Combining NBN-style local updates with transformer architecture innovations could yield novel training paradigms.

4. **Fine-Tuning Applications**: Second-order methods (HiZOO) show particular promise for LLM fine-tuning where full training is not needed.

---

## Conclusion

The Neuron-by-Neuron training algorithm pioneered by Dr. Wilamowski contains several forward-thinking concepts that resonate with cutting-edge LLM research:

- **Forward-only computation** is being actively explored as an alternative to backpropagation
- **Per-neuron/layer training** enables parallelism and memory efficiency
- **Second-order optimization** delivers significant convergence speedups when properly approximated
- **Arbitrary connectivity** supports modern architectures like transformers with skip connections

While direct application of NBN to billion-parameter LLMs faces computational challenges, the underlying principles—local learning, forward computation, and efficient curvature estimation—are at the forefront of modern deep learning research. The pyNBN implementation serves as a valuable testbed for these concepts and could inform future developments in efficient neural network training.

---

## References

### Original NBN Algorithm Papers

1. **Wilamowski, B.M., Yu, H.** "Neural Network Learning Without Backpropagation." *IEEE Transactions on Neural Networks*, 2010.
   - [SciSpace](https://scispace.com/papers/neural-network-learning-without-backpropagation-qxqobu4khr)

2. **Wilamowski, B.M., et al.** "NBN Algorithm." Chapter 13 in *Intelligent Systems*, CRC Press, 2011.
   - [Auburn University PDF](https://www.eng.auburn.edu/~wilambm/pap/2011/K10149_C013.pdf)
   - [Taylor & Francis](https://www.taylorfrancis.com/chapters/edit/10.1201/9781315218427-13/nbn-algorithm-bogdan-wilamowski-hao-yu-nicholas-cotton)

3. **Wilamowski, B.M., et al.** "Efficient and Reliable Training of Neural Networks."
   - [IEEE Xplore](https://ieeexplore.ieee.org/abstract/document/5090963)

4. **Wilamowski, B.M.** "Neural Network Architectures and Learning." Tutorial.
   - [INRIA Tutorial PDF](https://mistis.inrialpes.fr/docs/biblioPlaneto/Tutorial_Wilamowski.pdf)

### Forward-Only and Backpropagation-Free Training

5. **Li, Q., Teh, Y.W., Pascanu, R.** "NoProp: Training Neural Networks without Full Back-propagation or Full Forward-propagation." arXiv:2503.24322, 2025.
   - [arXiv](https://arxiv.org/abs/2503.24322)
   - [GitHub Implementation](https://github.com/Sid3503/NoProp)

6. **"Mono-Forward: Backpropagation-Free Algorithm for Efficient Neural Network Training Harnessing Local Errors."** 2024.
   - [HuggingFace Papers](https://huggingface.co/papers/2501.09238)

7. **Hinton, G.** "The Forward-Forward Algorithm: Some Preliminary Investigations." arXiv:2212.13345, 2022.
   - [arXiv PDF](https://arxiv.org/pdf/2212.13345)
   - [Toronto CS](https://www.cs.toronto.edu/~hinton/FFA13.pdf)

8. **"A Survey of Backpropagation-Free Training For LLMs."** Beijing University of Posts and Telecommunications, 2024.
   - [Survey PDF](https://d197for5662m48.cloudfront.net/documents/publicationstatus/201426/preprint_pdf/883fb6f1ba80815ebf0e357734d27f40.pdf)

### Second-Order Optimization

9. **Abreu, R., et al.** "The Potential of Second-Order Optimization for LLMs: A Study with Full Gauss-Newton." arXiv:2510.09378, 2025.
   - [arXiv](https://arxiv.org/abs/2510.09378)
   - [OpenReview PDF](https://openreview.net/pdf?id=yxEop1S5le)

10. **"Second-Order Fine-Tuning without Pain for LLMs: A Hessian Informed Zeroth-Order Optimizer (HiZOO)."** ICLR 2025.
    - [OpenReview](https://openreview.net/forum?id=bEqI61iBue)

11. **"Improving Levenberg-Marquardt Algorithm for Neural Networks."** arXiv:2212.08769, 2022.
    - [arXiv](https://arxiv.org/abs/2212.08769)

12. **"torch-levenberg-marquardt"** - PyTorch Implementation.
    - [GitHub](https://github.com/fabiodimarco/torch-levenberg-marquardt)

### Layer-Wise and Progressive Training

13. **Belilovsky, E., et al.** "Greedy Layerwise Learning Can Scale to ImageNet." ICML 2019.
    - [PMLR PDF](https://proceedings.mlr.press/v97/belilovsky19a/belilovsky19a.pdf)

14. **"GLNAS: Greedy Layer-wise Network Architecture Search."** Pattern Recognition, 2024.
    - [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0031320324004813)

15. **"GrowNN: Growing with Experience in Deep Reinforcement Learning."** arXiv:2506.11706, 2025.
    - [arXiv](https://arxiv.org/abs/2506.11706)

### Sparse and Arbitrary Connectivity

16. **"Brain network science modelling of sparse neural networks enables scalable training."** arXiv:2501.19107, 2025.
    - [arXiv](https://arxiv.org/abs/2501.19107)
    - [ICLR 2025](https://iclr.cc/virtual/2025/33513)

17. **"Training very deep neural networks: Rethinking the role of skip connections."** Neurocomputing, 2021.
    - [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0925231221002332)

18. **"Why Is Everyone Training Very Deep Neural Network With Skip Connections?"** IEEE Access, 2021.
    - [IEEE Xplore](https://ieeexplore.ieee.org/document/9669990)

### Mixture of Experts and Transformer Architectures

19. **"Mixture-of-Experts in the Era of LLMs."** University of Washington Course Notes, 2024.
    - [Course PDF](https://courses.cs.washington.edu/courses/cse599k/24au/content/MoE.pdf)

20. **HuggingFace.** "Large Language Model Training Playbook - Model Architecture."
    - [DeepWiki](https://deepwiki.com/huggingface/large_language_model_training_playbook/2-model-architecture)

### Forward-Forward Implementations

21. **"pytorch_forward_forward"** - PyTorch Implementation of Hinton's Forward-Forward Algorithm.
    - [GitHub](https://github.com/mpezeshki/pytorch_forward_forward)

22. **"loeweX/Forward-Forward"** - Reimplementation of the Forward-Forward Algorithm.
    - [GitHub](https://github.com/loeweX/Forward-Forward)

23. **"On Advancements of the Forward-Forward Algorithm."** arXiv:2504.21662, 2025.
    - [arXiv](https://arxiv.org/html/2504.21662v1)

---

*Document prepared for the pyNBN project. Last updated: December 2024.*
