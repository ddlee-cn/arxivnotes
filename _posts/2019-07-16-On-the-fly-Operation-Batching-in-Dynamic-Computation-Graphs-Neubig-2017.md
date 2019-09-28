---
title: "On-the-fly Operation Batching in Dynamic Computation Graphs - Neubig et al. - 2017"
tag:
- Engineering
---



## Info
- Title: **On-the-fly Operation Batching in Dynamic Computation Graphs**
- Author: Graham Neubig, Yoav Goldberg, Chris Dyer
- Arxiv: [1705.07860](https://arxiv.org/abs/1705.07860)
- Date: May. 2017

## Highlights & Drawbacks
Batch computation on dynamic graph

## Motivation & Design

Dynamic learning-based deep learning frameworks such as `Pytorch`, `DyNet` provide a more flexible choice of structure and data dimensions, but require developers to batchize the data themselves to maximize the parallel computing advantages of the framework.

### Current situation: flexible structure and efficient calculation

![On-the-fly Operation Batching in Dynamic Computation Graphs](https://i.imgur.com/gcArtSK.png)

The left picture shows the loop structure, and the right picture completes the sequence, batching

1. Flexible structure and data input dimensions, implemented in a simple loop structure, but not efficient, because the data accepts the same operation in the loop despite the different dimensions.
![On-the-fly Operation Batching in Dynamic Computation Graphs](https://i.imgur.com/RIf0zb8.png)


2. “Padding” the data, that is, using the data to align the input dimensions to achieve vectorization, but this operation is not friendly to the developer, which will cause the developer to waste a lot of investment in structural design. energy.

![On-the-fly Operation Batching in Dynamic Computation Graphs](https://i.imgur.com/JBpQnrx.png)


### Proposed Methods
1. Graph Definition
2. Operation Batching
3. Computation

The first and third steps are currently well implemented by most deep learning frameworks. The main feature is to construct the separation of the calculation graph from the calculation, namely "Lazy Evaluation". For example, in `Tensorflow`, an abstraction layer is responsible for parsing the dependencies between the nodes of the computation graph, determining the order in which the calculations are performed, and another abstraction layer is responsible for allocating computational resources.

#### Computing compatibility groups
This step is to build a group of nodes that can be batched. The specific approach is to create *signature* for each compute node to describe the characteristics of the node calculation. The following examples are given:

1. Component-wise operations: Calculations applied directly to each tensor element, regardless of the dimension of the tensor, such as $tanh$, $log$
2. Dimension-sensitive operations: Dimension-based calculations, such as linear transfer of $Wh+b$, require $W$ and $h$ dimensions to match, and *signature* to include dimension information
3. Operations with shared elements: Calculations that contain shared elements, such as shared weights $W$
4. Unbatchable operations: other

#### Determining execution order
The order of execution has to meet two goals:
1. The calculation of each node is after its dependence
2. Nodes with the same *signature* and no dependencies are placed in the same batch execution

However, in general, finding the order of execution to maximize the batch size is an NP problem. There are two strategies as follows:

1. Depth-based Batching: The method used in the library `Tensorflow Fold`. The depth of a node is defined as the maximum length of its child nodes to itself, and nodes at the same depth are bulk calculated. However, due to the length of the input sequence, some opportunities for batching may be missed.
2. Agenda-based Batching: The core idea of ​​this paper is to maintain a *agenda* sequence, all nodes that depend on the parsed have been parsed, and each iteration is taken from the *agenda* sequence by the same principle as *signature* The node performs batch calculations.


## Experiment & Ablation Study
The paper selected four models: BiLSTM, BiLSTM w/char, Tree-structured LSTMs, Transition-based Dependency Parsing.

Experimental results: (unit is Sentences/second)
![On-the-fly Operation Batching in Dynamic Computation Graphs](https://i.imgur.com/aQIZcPE.png)


