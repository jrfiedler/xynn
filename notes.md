# Notes


## Misc

- AutoInt
  - paper makes the claim that 0.001 improvement in log loss is regarded as significant for CTR
  - official implementation uses ReLU activations on its W matrices (named `Q`, `K`, `V`, and `V_res`); why?
    - it doesn't seem to matter much, but there's a parameter attn_activation to control this


## Models

### Initialization

- per-model standards, or one standard across models?
- what did each model do?

### Embedding

- linear embedding, default embedding

- PNN
  - did not consider non-categorical fields [check the implementations]
  - used basic embedding? [check the implementations]
  - vector length 10 (see RHS of pg 4)
  -
- xDeepFM
- AutoInt
  - embeds both numeric and categorical fields
  - uses linear embedding for numeric
  - multi-valued features use the element-wise mean of each value's embedding
  - in the paper, the analysis of space complexity suggests that only the basic type of categorical embedding was considered (lower RHS of page 5)
- FiBiNet
- TabNet

### MLP

- individual defaults or one default across models? (especially, hidden sizes)
- leaky gate

#### Activations

- PNN
  - paper compared performance of ReLU, tanh, and sigmoid
  - paper used ReLU
  - "more detailed implementation" allowed more types of activation
- AutoInt
  - official implementation uses ReLU
- here: LeakyReLU default, other activations can be specified by user
- sparsemax and entmax15

#### Batch Normalization and Dropout

The default here is to use batch normalization with no dropout in MLP linear layers. [Need to change code for this.]
- PNN used dropout and concluded that 0.5 was an effective default.
- xDeepFM
- AutoInt
  - the paper mentions searching over dropout values (top left paragraph, page 7), but the official implmentation allows the use of batch norm in addition to dropout
- FiBiNet
- TabNet

#### Side MLPs

- which already had this, which did not?
  - PNN does not
  - AutoInt did not, but "AutoInt+" did
  - xDeepFM did
  - FiBiNet did not
  - TabNet did not
  - TabTransformer did not
- weighted sum


### PNN

Paper: https://arxiv.org/pdf/1611.00144v1.pdf<br>
Official implementation: https://github.com/Atomu2014/product-nets<br>
"More detailed" implementation: https://github.com/Atomu2014/product-nets-distributed/<br>
Other implementations:
- https://github.com/JianzhouZhan/Awesome-RecSystem-Models
- https://github.com/shenweichen/DeepCTR-Torch

#### Changes from the paper / official implementation


#### Other notes
- The implementation of products follows JianzhouZhan's implementation
- The paper seems to indicate that linear and product outputs are summed, with a bias vector, before being transformed through the MLP (eq 3, page 2). However, in the official implementation (and other implementations) the linear and product outputs are concatenated.
- The paper uses names IPNN, OPNN, PNN\*, which only differ by the type of product. We use the name "PNN" for all of these.
- Embedding
  - At the beginning of section III, the paper only considers categorical features, and the paper does not mention the use of non-categorical features.
  - [Does the iPinYou dataset have numeric features?]
  - [How do the other reference implementation handle embedding?]
- Like many NN CTR models, the only application considered is CTR. In particular, the paper only considers a single output in the range (0, 1) (see eq. 1 and first paragraph under Fig 1, page 2). Here, we allow multi-class classification and (multiple) regression.
- The models here don't transform classification values to the range (0, 1), leaving that to the loss function or downstream application code.
- The paper uses ReLU activations (section III.A., eq 2 and eq 3). We allow user-specificed activations, with LeakyReLU as default.
- The paper only considers log loss (final paragraph of section III.A., page 3). We allow user-specified losses.
- [Dropout and batch normalization?]


### AutoInt

Paper: https://arxiv.org/pdf/1810.11921v2.pdf<br>
Official implementation: https://github.com/DeepGraphLearning/RecommenderSystems<br>


### xDeepFM


### FiBiNet


### TabNet


### TabTransformer


### New models


## Data

### Rossman

Are all non-target fields usable? Should we be dropping Customers? Check original processing and Kaggle.

### Forest cover

### Higgs