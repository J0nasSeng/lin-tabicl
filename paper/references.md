# Related Work — Verified References

Every entry below was confirmed against a fetched source page (DBLP, arXiv API, PMLR, or Semantic Scholar). Published versions are reported in preference to preprints. Entries marked *(preprint)* have no published version found.

Grouped by the manuscript section they support: `paper/iclr2027_conference.tex`.

---

## 1. Introduction — Tabular foundation models and in-context learning

Transformers Can Do Bayesian Inference. Samuel Müller, Noah Hollmann, Sebastian Pineda-Arango, Josif Grabocka, Frank Hutter. 2022. Introduces Prior-Data Fitted Networks (PFNs), which learn to approximate Bayesian inference by pre-training on synthetic datasets drawn from a prior. This is the foundational formalism behind the "pre-train on millions of synthetic datasets" claim in the opening paragraph.

TabPFN: A Transformer That Solves Small Tabular Classification Problems in a Second. Noah Hollmann, Samuel Müller, Katharina Eggensperger, Frank Hutter. 2023. The first tabular foundation model, performing classification by in-context learning in a single forward pass with no gradient updates at test time. The canonical citation for the TFM paradigm the paper builds on.

Accurate predictions on small data with a tabular foundation model. Noah Hollmann, Samuel Müller, Lennart Purucker, Arjun Krishnakumar, Max Körfer, Shi Bin Hoo, Robin Tibor Schirrmeister, Frank Hutter. 2025. The TabPFN v2 journal version, extending the approach to regression and larger, more heterogeneous datasets. Supports the claim that TFMs achieve remarkable performance across classification and regression.

TabICL: A Tabular Foundation Model for In-Context Learning on Large Data. Jingang Qu, David Holzmüller, Gaël Varoquaux, Marine Le Morvan. 2025. A tabular foundation model explicitly targeting scalability to large datasets, using a two-stage column-then-row attention architecture. Directly relevant to both the architecture and the many-rows cost argument.

TabICLv2: A better, faster, scalable, and open tabular foundation model. Jingang Qu, David Holzmüller, Gaël Varoquaux, Marine Le Morvan. 2026. *(preprint)* The reference model whose data prior is reused unmodified in Section 2.3 and whose training recipe is followed in Section 2.4. Essential citation for the fixed-prior comparison rationale.

Why do tree-based models still outperform deep learning on typical tabular data? Léo Grinsztajn, Edouard Oyallon, Gaël Varoquaux. 2022. Benchmarks tree-based models against deep networks on tabular data and analyses their differing inductive biases. Standard reference for why tabular learning is treated as its own modality, and a natural baseline framing for the evaluation.

## 2. Introduction — Quadratic cost of dense attention and efficient alternatives

Attention is All you Need. Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin. 2017. Introduces the Transformer and the dense self-attention operation whose quadratic scaling in sequence length is the cost the paper seeks to avoid.

Big Bird: Transformers for Longer Sequences. Manzil Zaheer, Guru Guruganesh, Avinava Dubey, Joshua Ainslie, Chris Alberti, Santiago Ontanon, Philip Pham, Anirudh Ravula, Qifan Wang, Li Yang, Amr Ahmed. 2020. A sparse attention mechanism that reduces the quadratic dependency to linear while remaining a universal approximator. The closest precedent for the paper's core claim that dense attention can be sparsified without losing capability.

Longformer: The Long-Document Transformer. Iz Beltagy, Matthew E. Peters, Arman Cohan. 2020. *(preprint)* Combines local windowed attention with task-motivated global attention to scale linearly in sequence length. Relevant precedent for restricting which elements may exchange information.

Rethinking Attention with Performers. Krzysztof Choromanski, Valerii Likhosherstov, David Dohan, Xingyou Song, Andreea Gane, Tamas Sarlos, Peter Hawkins, Jared Davis, Afroz Mohiuddin, Lukasz Kaiser, David Belanger, Lucy Colwell, Adrian Weller. 2021. Achieves linear-time attention by kernel approximation rather than sparsity. Useful contrast: the paper attains linear cost through an explicit sparse graph instead of a low-rank or kernel approximation.

Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks. Juho Lee, Yoonho Lee, Jungtaek Kim, Adam Kosiorek, Seungjin Choi, Yee Whye Teh. 2019. Reduces set self-attention from quadratic to linear using inducing points, and provides the permutation-invariant set formulation appropriate for datasets as inputs. The inducing-point mechanism corresponds to the column embedder configuration in Section 2.4.

FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness. Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, Christopher Ré. 2022. An IO-aware exact attention algorithm that reduces memory traffic without changing the attention pattern. Important contrast for the paper's argument, and the mechanism enabled in training stages 2 and 3.

## 3. Method §2.2 — Architecture: message passing, positional encoding, readout

Graph Attention Networks. Petar Veličković, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Liò, Yoshua Bengio. 2018. Introduces attention-based neighbourhood aggregation on graphs. The direct architectural basis for the GAT backbone used as the ICL mechanism.

How Attentive are Graph Attention Networks? Shaked Brody, Uri Alon, Eran Yahav. 2022. Shows the original GAT computes only static attention and proposes GATv2 with dynamic attention. Relevant to justifying the specific attention formulation in Equation 1.

Neural Message Passing for Quantum Chemistry. Justin Gilmer, Samuel S. Schoenholz, Patrick F. Riley, Oriol Vinyals, George E. Dahl. 2017. Unifies graph neural networks under the Message Passing Neural Network framework. The standard citation for the message-passing formalism the paper adopts.

Semi-Supervised Classification with Graph Convolutional Networks. Thomas N. Kipf, Max Welling. 2017. Establishes the semi-supervised node classification setting with a labelled subset and unlabelled targets in one graph. Structurally the same problem as the support/query formulation in Section 2.1.

Inductive Representation Learning on Large Graphs. William L. Hamilton, Rex Ying, Jure Leskovec. 2017. Introduces neighbour sampling for inductive node embedding on large graphs. Directly relevant to the bounded-degree sampling used to keep message passing linear.

RoFormer: Enhanced transformer with Rotary Position Embedding. Jianlin Su, Murtadha H. M. Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, Yunfeng Liu. 2024. Introduces rotary position embeddings (RoPE) encoding relative position through rotation. Cited for the RoPE configuration of the row interactor in Section 2.4.

Scalable-Softmax Is Superior for Attention. Ken M. Nakanishi. 2025. *(preprint)* Replaces softmax with a size-aware variant that prevents attention from flattening as context size grows. Corresponds to the scalable-softmax setting used in the column embedder and ICL backbone.

Prototypical Networks for Few-shot Learning. Jake Snell, Kevin Swersky, Richard S. Zemel. 2017. Classifies query points by distance to class prototypes in a learned metric space. The closest precedent for the distance-based soft-kNN readout in Equation 3.

## 4. Method §2.3 — Graph construction, homophily, and oversmoothing

Beyond Homophily in Graph Neural Networks: Current Limitations and Effective Designs. Jiong Zhu, Yujun Yan, Lingxiao Zhao, Mark Heimann, Leman Akoglu, Danai Koutra. 2020. Shows that many GNNs degrade under heterophily and identifies designs that help. Directly relevant to the "homophilic" scope in the title and to the class-conditioned edge construction with a cross-class fraction.

Geom-GCN: Geometric Graph Convolutional Networks. Hongbin Pei, Bingzhe Wei, Kevin Chen-Chuan Chang, Yu Lei, Bo Yang. 2020. Proposes a geometric aggregation scheme addressing MPNN weaknesses on disassortative graphs. Supports the homophily/heterophily framing and supplies standard node classification datasets.

PairNorm: Tackling Oversmoothing in GNNs. Lingxiao Zhao, Leman Akoglu. 2020. Introduces a normalization layer preventing node embeddings from becoming indistinguishable in deep GNNs. Direct support for the oversmoothing concern raised in the Section 2.4 model configuration.

DropEdge: Towards Deep Graph Convolutional Networks on Node Classification. Yu Rong, Wenbing Huang, Tingyang Xu, Junzhou Huang. 2020. Randomly removes edges each epoch, acting as both data augmentation and a message-passing reducer that provably slows oversmoothing. The closest precedent for resampling several graphs per forward pass as a regularizer.

## 5. Method §2.4 — Training and optimization

Decoupled Weight Decay Regularization. Ilya Loshchilov, Frank Hutter. 2019. Introduces AdamW by decoupling weight decay from the gradient update. Relevant to the weight-decay formulation of the optimizer used in training.

Muon is Scalable for LLM Training. Jingyuan Liu, Jianlin Su, Xingcheng Yao, Zhejun Jiang, Guokun Lai, Yulun Du, Yidao Qin, Weixin Xu, Enzhe Lu, Junjie Yan, and others. 2025. *(preprint)* Documents the Muon matrix-orthogonalization optimizer at scale, including weight decay and update-scale adjustments. The citable reference for the Muon optimizer named in Section 2.4.

## 6. Graph in-context learning and node classification transfer

PRODIGY: Enabling In-context Learning Over Graphs. Qian Huang, Hongyu Ren, Peng Chen, Gregor Kržmanc, Daniel Zeng, Percy Liang, Jure Leskovec. 2023. The first pretraining framework for in-context learning over graphs, connecting prompt examples and queries through a prompt-graph representation. The most direct prior work for conditioning a model on an explicit graph linking labelled and unlabelled nodes.

Open Graph Benchmark: Datasets for Machine Learning on Graphs. Weihua Hu, Matthias Fey, Marinka Zitnik, Yuxiao Dong, Hongyu Ren, Bowen Liu, Michele Catasta, Jure Leskovec. 2020. Provides large-scale graph datasets with unified evaluation protocols, including node classification. The standard benchmark source for the claimed transfer to node classification.
