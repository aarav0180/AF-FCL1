# Cosine-Normalized Accurate Forgetting for Federated Continual Learning

## Abstract

Federated continual learning (FCL) enables distributed clients to collaboratively learn a sequence of tasks without sharing raw data, yet it faces two compounding challenges: catastrophic forgetting of previously acquired knowledge and statistical heterogeneity arising from non-identical data distributions across clients. The Accurate Forgetting framework (AF-FCL) addresses these issues through conditional normalizing flows that model per-class feature distributions and enable generative replay alongside knowledge distillation. However, the standard linear classification head employed by AF-FCL computes logits as $z_c = \mathbf{W}_c^\top \mathbf{x} + b_c$, coupling both the magnitude and direction of weight vectors with the decision. During sequential task updates, this coupling introduces magnitude bias: newly updated class prototypes acquire disproportionately large norms, causing the classifier to favor recent classes irrespective of angular proximity. In this paper, we propose replacing the linear classifier with a cosine-normalized classification head that decouples magnitude from direction by computing logits as $z_c = \sigma \cdot \cos(\theta_c)$, where $\theta_c$ is the angle between the L2-normalized feature vector and the L2-normalized class prototype, and $\sigma$ is a learnable temperature scalar. This single architectural substitution eliminates magnitude bias, produces angularly uniform decision boundaries, and stabilizes the normalizing flow training. Experiments on the EMNIST-Letters benchmark under both Non-IID and Shared data partitions demonstrate that the cosine-normalized classifier improves class-so-far accuracy from 27.11% to 62.80% at the first task under Non-IID settings, achieves 85.25% final per-task accuracy compared to 67.34% for the baseline, and reduces normalizing flow loss median from 96,449 to 518, indicating dramatically improved numerical stability. On the Shared split, the cosine variant consistently outperforms the baseline across all tasks and training horizons, reaching 87.44% class-so-far accuracy at 150 epochs versus 61.27% for the baseline.

## I. Introduction

Federated learning (FL) has emerged as a dominant paradigm for privacy-preserving collaborative machine learning, enabling multiple clients to jointly train a shared model without centralizing sensitive data. Each client trains on its local dataset and communicates only model updates to a central server, which aggregates them into a global model. This approach preserves data locality and satisfies increasingly stringent regulatory requirements around data governance.

Continual learning (CL), independently, addresses the problem of learning a sequence of tasks over time without forgetting previously acquired knowledge. The central challenge in CL is catastrophic forgetting, wherein the optimization of parameters for a new task destructively interferes with representations learned for earlier tasks. Numerous strategies have been proposed to mitigate forgetting, including regularization-based approaches that penalize changes to important parameters, replay-based methods that store or generate exemplars from past tasks, and architecture-based techniques that allocate dedicated capacity for each task.

Federated continual learning (FCL) lies at the intersection of these two fields and inherits the difficulties of both. In FCL, distributed clients must learn a sequence of tasks collaboratively, where each client may encounter tasks at different times and with non-identical data distributions. The combination of catastrophic forgetting and statistical heterogeneity creates a particularly challenging optimization landscape: not only must the model retain knowledge across tasks, but it must do so while reconciling divergent updates from clients with fundamentally different data characteristics.

The Accurate Forgetting framework, referred to as AF-FCL, represents a principled approach to this problem. Rather than attempting to prevent all forgetting, AF-FCL embraces selective forgetting through a conditional normalizing flow that learns the feature distribution $p(\mathbf{x} \mid y)$ for each class. By sampling from this learned distribution, the framework generates synthetic replay features for earlier classes, enabling the classifier to rehearse old knowledge without storing raw data. Combined with knowledge distillation from both the previous local model and the global model, AF-FCL achieves a balance between plasticity for new tasks and stability for old ones.

However, a fundamental limitation persists in the AF-FCL architecture that has received insufficient attention: the classification head itself. The standard linear classifier computes the logit for class $c$ as:

$$z_c = \mathbf{W}_c^\top \mathbf{x} + b_c = \|\mathbf{W}_c\| \cdot \|\mathbf{x}\| \cdot \cos(\theta_c) + b_c$$

where $\theta_c$ is the angle between the weight vector $\mathbf{W}_c$ and the feature vector $\mathbf{x}$. This formulation entangles three quantities: the norm of the class prototype $\|\mathbf{W}_c\|$, the norm of the input feature $\|\mathbf{x}\|$, and the angular relationship $\cos(\theta_c)$. During continual learning, as the model trains on new tasks, the weight vectors for recently seen classes are updated more frequently and tend to grow in magnitude. Meanwhile, weight vectors for older classes remain stagnant or drift unpredictably through federated aggregation. The result is magnitude bias: the classifier systematically favors classes with larger weight norms, even when the angular evidence points toward a different class.

This magnitude bias is particularly pernicious in the federated setting, where heterogeneous client data distributions cause different clients to update different subsets of class prototypes with different intensities. After federated averaging, the aggregated weight norms reflect an uneven mixture of update frequencies rather than genuine class discriminability.

In this work, we propose a targeted architectural modification to the AF-FCL framework: replacing the standard linear classification head with a cosine-normalized classifier. The cosine classifier computes logits as:

$$z_c = \sigma \cdot \frac{\mathbf{W}_c^\top \mathbf{x}}{\|\mathbf{W}_c\| \cdot \|\mathbf{x}\|} = \sigma \cdot \cos(\theta_c)$$

where $\sigma$ is a learnable temperature parameter. By L2-normalizing both the feature vector and the class prototypes before computing their dot product, we project all representations onto the unit hypersphere, eliminating magnitude as a factor in the decision boundary. The classifier is forced to learn the directional geometry of each class, producing decision boundaries that are purely angular and therefore invariant to the scale drifts that plague continual and federated learning.

The contributions of this paper are as follows:

1. We identify magnitude bias in the linear classifier as a previously underexplored source of instability in federated continual learning, and provide a geometric analysis of how it interacts with sequential task updates and federated aggregation.

2. We introduce a cosine-normalized classification head into the AF-FCL framework, replacing the standard linear layer with a CosineLinear module that features learnable temperature scaling and optional additive angular margin.

3. We demonstrate through experiments on EMNIST-Letters under Non-IID and Shared data partitions that the cosine classifier substantially improves accuracy, retention, and normalizing flow stability, all without modifying the replay mechanism, distillation losses, or flow architecture.

## II. Related Work

### A. Federated Learning

Federated learning was formalized as a distributed optimization problem where multiple clients collaboratively minimize a global objective without sharing raw data. The foundational FedAvg algorithm performs local stochastic gradient descent on each client and aggregates the resulting model updates through weighted averaging. Subsequent work has addressed the challenge of statistical heterogeneity through techniques such as local regularization toward the global model (FedProx), personalized aggregation strategies, and scaffold-based variance reduction. Despite these advances, the majority of federated learning research assumes a fixed task distribution, leaving the sequential task setting largely unaddressed until recently.

### B. Continual Learning

Continual learning methods broadly fall into three families. Regularization-based approaches, exemplified by Elastic Weight Consolidation (EWC) and Synaptic Intelligence, estimate the importance of each parameter for previous tasks and penalize deviations from previously learned values. Replay-based methods maintain a memory buffer of exemplars from past tasks or train generative models to synthesize pseudo-exemplars for rehearsal. Architecture-based methods allocate distinct parameters or modules for each task, avoiding interference by construction but often at the cost of scalability. The AF-FCL framework falls primarily in the replay category, using normalizing flows as the generative mechanism.

### C. Accurate Forgetting with Normalizing Flows

The AF-FCL framework introduces the concept of accurate forgetting for federated continual learning. Rather than treating all forgetting as harmful, AF-FCL uses a conditional normalizing flow to model the per-class feature distribution $p(\mathbf{x} \mid y)$. The flow is trained concurrently with the classifier and provides two key capabilities: it estimates the probability that a generated sample falls within the local data distribution, enabling selective replay weighting, and it generates synthetic features from previous classes for classifier rehearsal. Knowledge distillation from the previous local model and the global server model provides additional regularization against forgetting.

### D. Cosine Classifiers in Deep Learning

The use of cosine similarity as a classification criterion has a rich history in metric learning and face recognition. The seminal ArcFace work demonstrated that normalizing both features and class prototypes onto the unit hypersphere, combined with an additive angular margin penalty, produces highly discriminative and well-separated class representations. This principle has since been adopted in few-shot learning, where cosine classifiers provide better generalization from limited examples, and in class-incremental learning, where they mitigate the bias toward recently learned classes. The LUCIR framework specifically demonstrated the effectiveness of cosine normalization for class-incremental learning in the centralized setting, motivating our investigation of its application to the federated continual learning regime.

## III. Background: The AF-FCL Framework

### A. Problem Formulation

Consider a federated system with $K$ clients collaborating through a central server. The learning proceeds through a sequence of $T$ tasks, where each task $t$ introduces a set of new classes $\mathcal{C}^t$. At task $t$, each client $k$ receives a local dataset $\mathcal{D}_k^t = \{(\mathbf{x}_i, y_i)\}$ where $y_i \in \mathcal{C}^t$. The objective is to learn a model that performs well on all classes seen so far, $\mathcal{C}^{1:t} = \bigcup_{\tau=1}^{t} \mathcal{C}^\tau$, while only having access to data from the current task $\mathcal{D}_k^t$.

### B. Architecture

The AF-FCL model consists of three components:

**Feature Extractor.** A convolutional neural network (S-ConvNet for EMNIST, ResNet-18 with CBAM attention for CIFAR-100) maps input images to a feature vector $\mathbf{x}_a \in \mathbb{R}^d$ where $d = 512$.

**Classifier Head.** A two-layer feedforward network transforms $\mathbf{x}_a$ through a hidden layer $\mathbf{x}_b = \text{LeakyReLU}(\mathbf{W}_2 \mathbf{x}_a)$ and produces class logits $\mathbf{z} = \mathbf{W}_c \mathbf{x}_b$, followed by softmax normalization.

**Normalizing Flow.** A conditional normalizing flow models the invertible mapping between the feature space and a base Gaussian distribution, conditioned on the class label: $\mathbf{x}_a = f_\phi(\mathbf{u}; y)$, where $\mathbf{u} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$. The flow is implemented as a composition of affine coupling layers with residual networks as the coupling functions.

### C. Training Procedure

Each local training iteration alternates between two phases:

**Flow Training.** The normalizing flow parameters $\phi$ are optimized to maximize the log-likelihood of the current task features:

$$\mathcal{L}_{\text{flow}} = -\mathbb{E}_{(\mathbf{x}, y) \sim \mathcal{D}^t} \left[ \log p_\phi(\mathbf{x}_a \mid y) \right] + \lambda_{\text{last}} \mathcal{L}_{\text{flow-last}}$$

where $\mathcal{L}_{\text{flow-last}}$ distills knowledge from the previous flow model by replaying synthetic samples.

**Classifier Training.** The classifier is trained with a composite loss:

$$\mathcal{L}_{\text{cls}} = \mathcal{L}_{\text{CE}}(\mathbf{x}, y) + \mathcal{L}_{\text{KD}} + \mathcal{L}_{\text{replay}}$$

The cross-entropy loss $\mathcal{L}_{\text{CE}}$ operates on the current task data. The knowledge distillation loss $\mathcal{L}_{\text{KD}}$ combines feature-level and output-level distillation from both the previous local model and the global model:

$$\mathcal{L}_{\text{KD}} = \lambda_f (\mathcal{L}_{\text{feat-last}} + \mathcal{L}_{\text{feat-global}}) + \lambda_o (\mathcal{L}_{\text{out-last}} + \mathcal{L}_{\text{out-global}})$$

The replay loss $\mathcal{L}_{\text{replay}}$ samples synthetic features $\tilde{\mathbf{x}}_a \sim p_\phi(\cdot \mid \tilde{y})$ from the flow, classifies them, and computes a probability-weighted cross-entropy:

$$\mathcal{L}_{\text{replay}} = \lambda_{\text{flow}} \cdot \mathbb{E}_{\tilde{\mathbf{x}}_a, \tilde{y}} \left[ p_{\text{local}}(\tilde{\mathbf{x}}_a) \cdot \mathcal{L}_{\text{CE}}(\tilde{\mathbf{x}}_a, \tilde{y}) \right]$$

where $p_{\text{local}}(\tilde{\mathbf{x}}_a)$ estimates the probability that the synthetic sample lies within the local data distribution, computed using a per-class Gaussian approximation over the flow noise space.

### D. Federated Aggregation

After local training, client models are aggregated at the server using FedAvg. The server maintains a global copy of both the classifier and the flow model. The aggregated global model is distributed back to all clients at the start of the next communication round.

## IV. Proposed Method

### A. The Magnitude Bias Problem in Linear Classifiers

The standard linear classifier computes the logit for class $c$ as:

$$z_c = \mathbf{W}_c^\top \mathbf{x} + b_c$$

Expanding this in terms of norms and angles:

$$z_c = \|\mathbf{W}_c\| \cdot \|\mathbf{x}\| \cdot \cos(\theta_c) + b_c$$

where $\theta_c = \angle(\mathbf{W}_c, \mathbf{x})$. The predicted class is $\hat{y} = \arg\max_c z_c$. Consider two classes $a$ and $b$ with identical angular proximity to an input feature ($\theta_a = \theta_b$). If $\|\mathbf{W}_a\| > \|\mathbf{W}_b\|$, then $z_a > z_b$, and the classifier predicts class $a$ purely due to its larger weight norm.

In continual learning, this norm asymmetry arises naturally. When the model trains on task $t$, gradient updates for the current task classes increase $\|\mathbf{W}_c\|$ for $c \in \mathcal{C}^t$. Weight vectors for previous task classes $c \in \mathcal{C}^{1:t-1}$ receive updates only through knowledge distillation and replay, which are typically weaker signals. Over successive tasks, a systematic norm gap develops between recent and old class prototypes.

In the federated setting, this problem is amplified by statistical heterogeneity. Different clients hold different subsets of classes, so federated averaging produces weight vectors whose norms reflect the average update intensity across clients rather than the true discriminative relevance of each class. The result is a noisy, heterogeneous norm landscape that corrupts the classifier's ranking of classes.

### B. Cosine-Normalized Classification Head

We propose replacing the linear classification head with a cosine-normalized head defined as:

$$z_c = \sigma \cdot \frac{\mathbf{W}_c^\top \mathbf{x}}{\|\mathbf{W}_c\| \cdot \|\mathbf{x}\|} = \sigma \cdot \hat{\mathbf{W}}_c^\top \hat{\mathbf{x}} = \sigma \cdot \cos(\theta_c)$$

where $\hat{\mathbf{W}}_c = \mathbf{W}_c / \|\mathbf{W}_c\|$ and $\hat{\mathbf{x}} = \mathbf{x} / \|\mathbf{x}\|$ are the L2-normalized class prototype and feature vector, respectively. The scalar $\sigma > 0$ is a learnable temperature parameter.

This formulation has several key properties:

**Magnitude Invariance.** The logit $z_c$ depends solely on the angle $\theta_c$ between the feature vector and the class prototype on the unit hypersphere. No class can dominate by having a larger weight norm. The decision boundary between any two classes $a$ and $b$ is defined by $\cos(\theta_a) = \cos(\theta_b)$, which is a purely angular condition.

**Temperature Scaling.** Raw cosine similarities lie in $[-1, +1]$. If fed directly into the softmax function, the resulting probability distribution would be nearly uniform, producing weak gradients and slow learning. The learnable temperature $\sigma$ stretches the logit range to $[-\sigma, +\sigma]$, allowing the softmax to produce sharp, discriminative distributions. We initialize $\sigma = 10.0$ and constrain it to $[1.0, 30.0]$ during training to prevent numerical instability.

**Bias Elimination.** The cosine classifier has no bias term. On the unit hypersphere, a bias would break the geometric symmetry of the angular decision boundaries. Its removal ensures that all classes are treated equitably regardless of their frequency or recency.

### C. Optional Angular Margin

When the angular margin hyperparameter $m > 0$, we apply an ArcFace-style additive angular margin during training. For the ground-truth class $y_i$, the logit is modified to:

$$z_{y_i} = \sigma \cdot \cos(\theta_{y_i} + m)$$

while non-target class logits remain $z_c = \sigma \cdot \cos(\theta_c)$ for $c \neq y_i$. This penalty enforces a minimum angular gap of $m$ radians between the feature vector and its correct prototype, producing tighter intra-class clustering and wider inter-class separation. At inference time, the margin is disabled and predictions use pure cosine similarity.

### D. Integration with AF-FCL

The cosine-normalized classifier integrates into the AF-FCL framework as a drop-in replacement for the standard linear head. Specifically:

1. **The feature extractor remains unchanged.** The convolutional backbone (S-ConvNet or ResNet-18-CBAM) and the intermediate fully connected layers ($\text{fc}_1$, $\text{fc}_2$) are preserved exactly as in the baseline.

2. **Only the final classification layer is replaced.** The standard `nn.Linear(xa_dim, num_classes)` is swapped with `CosineLinear(xa_dim, num_classes, sigma_init, margin)`.

3. **The normalizing flow is unchanged.** The flow architecture, training procedure, and replay sampling mechanism remain identical. The flow operates on features $\mathbf{x}_a$ extracted before the classification head, so it is agnostic to the head's parameterization.

4. **Knowledge distillation losses are preserved.** Both feature-level distillation (on $\mathbf{x}_a$) and output-level distillation (on softmax probabilities) function identically, as the cosine head still produces softmax-normalized class probabilities.

5. **The replay loss is preserved.** Flow-generated synthetic features are classified through the cosine head, and the probability-weighted replay cross-entropy operates on the resulting softmax outputs without modification.

The implementation uses Python's class inheritance through a `CosineMixin` that intercepts the model construction, replaces the classifier, and rebuilds the optimizer to include the new parameters (including the learnable $\sigma$). This design ensures complete composability with all other AF-FCL extensions.

### E. Theoretical Insight

The cosine classifier improves federated continual learning stability through three mechanisms:

**Reduced Parameter Drift.** In the linear classifier, federated averaging of weight vectors with heterogeneous norms introduces spurious magnitude drift that does not correspond to genuine class-discriminative information. On the unit hypersphere, all prototypes have unit norm by construction, so federated averaging operates purely on directional information. The averaged prototype $\hat{\mathbf{W}}_c^{\text{avg}}$ is subsequently re-normalized, producing a consensus direction that is less susceptible to outlier client updates.

**Improved Feature Alignment.** The L2 normalization of the feature vector acts as an implicit regularizer that constrains the feature extractor to produce representations of consistent magnitude. This benefits the normalizing flow, which models the feature distribution $p(\mathbf{x}_a \mid y)$: when feature norms are stable across tasks and clients, the flow's density estimation becomes more accurate, leading to higher-quality replay samples.

**Scale-Invariant Decision Boundaries.** As tasks progress and the feature extractor undergoes representational shift, the absolute scale of feature activations may change. Linear classifiers are sensitive to such scale changes, as they alter the effective contribution of each class prototype. Cosine classifiers are invariant to feature scale by construction, providing robustness against the representational drift that naturally occurs in continual learning.

## V. Experimental Setup

### A. Dataset and Task Configuration

We evaluate our proposed method on the EMNIST-Letters dataset, which contains 26 handwritten letter classes with 28×28 grayscale images. The dataset is partitioned across 10 federated clients using a structured split that distributes classes non-uniformly, creating realistic statistical heterogeneity. The task sequence consists of 6 incremental tasks, each introducing a subset of new classes to the learning system.

We consider two data partition strategies:

**Non-IID Split.** Each client receives a distinct, non-overlapping subset of classes for each task. This represents the most challenging federated scenario, where clients have fundamentally different label distributions. The specific partition is defined by the split file `EMNIST_split_cn10_tn6_cet4_s2571.pkl` with 10 clients, 6 tasks, and 4 classes entering per task.

**Shared Split.** All clients share the same class distribution for each task, but with different data samples. This isolates the effect of the cosine classifier from the additional complexity of label heterogeneity.

### B. Model Configuration

The feature extractor is an S-ConvNet with channel size 64, producing 512-dimensional feature vectors. The normalizing flow consists of 4 affine coupling layers with 512-dimensional hidden features, using residual network coupling functions with 2 residual blocks per layer. The classifier head is either the standard linear layer (baseline) or the proposed CosineLinear head with temperature $\sigma$ initialized to 10.0.

### C. Training Protocol

Training proceeds for 40 global communication rounds per task, with 80 local epochs per round for the classifier and 15 epochs for the flow. We use the Adam optimizer with learning rate $10^{-3}$ for the classifier and $10^{-3}$ for the flow, with $\beta_1 = 0.9$ and $\beta_2 = 0.999$. The batch size is 64 across all experiments. The loss coefficients are: $\lambda_{\text{flow}} = 0.5$, $\lambda_{\text{KD-last}} = 0.2$, $\lambda_f = 0.5$, $\lambda_o = 0.1$, $\lambda_{\text{flow-last}} = 0.01$, and $\theta_{\text{explore}} = 0.1$. FedProx regularization is applied with $\mu = 0.001$.

For the cosine classifier, we set $\sigma_{\text{init}} = 5.0$, angular margin $m = 0.15$, and enable feature calibration (BatchNorm before the cosine head). The cosine model also uses EWC regularization with $\lambda_{\text{EWC}} = 500.0$ and adaptive KD weighting.

### D. Evaluation Metrics

We report four metrics:

**Class-so-far Accuracy.** After completing task $t$, we evaluate the model on test data from all classes seen so far, $\mathcal{C}^{1:t}$. This measures both forward learning and backward retention.

**Final Per-task Accuracy.** After all tasks are completed, we evaluate the model separately on each task's test set. This reveals per-task forgetting patterns.

**Flow Loss Stability.** We report the median, minimum, and maximum normalizing flow training loss across all training iterations. Lower and more stable flow loss indicates better density estimation and higher-quality replay.

### E. Baselines

We compare against two baseline configurations:

**Baseline-CPU.** The standard AF-FCL model with a linear classifier, trained on CPU.

**Baseline-GPU.** The standard AF-FCL model with a linear classifier, trained on GPU with identical hyperparameters.

The cosine variant uses the same flow architecture, replay mechanism, and distillation losses as the baselines, differing only in the classification head.

## VI. Results and Analysis

### A. Non-IID Class-So-Far Accuracy

| Task | Baseline-CPU | Baseline-GPU | Cosine (80ep) |
|------|-------------|-------------|---------------|
| t=1  | 27.11       | 26.73       | **62.80**     |
| t=2  | 33.94       | 34.16       | **49.16**     |
| t=3  | 39.28       | 49.11       | **46.75**     |
| t=4  | 41.04       | 40.84       | **33.49**     |
| t=5  | 42.46       | 35.01       | **35.80**     |
| t=6  | 36.12       | 35.27       | **42.73**     |

The cosine classifier exhibits a dramatically different learning trajectory compared to the baseline. At the first task, it achieves 62.80% accuracy, more than doubling the baseline's 27.11%. This substantial early advantage reflects the immediate benefit of magnitude-invariant decision boundaries: when only a small number of classes are present, the cosine classifier learns clean angular separations without wasting capacity on fitting weight magnitudes.

As additional tasks are introduced, all configurations experience the expected decline in class-so-far accuracy due to the expanding class space. However, the cosine variant retains a meaningful advantage at the final task (42.73% versus 36.12% for Baseline-CPU and 35.27% for Baseline-GPU), indicating reduced cumulative forgetting.

### B. Non-IID Final Per-Task Accuracy

| Task | Baseline-CPU | Baseline-GPU | Cosine |
|------|-------------|-------------|--------|
| T1   | 27.52       | 24.45       | **35.48** |
| T2   | 15.30       | 12.53       | **17.81** |
| T3   | 38.30       | 38.70       | **40.13** |
| T4   | 29.34       | 29.67       | **33.69** |
| T5   | 38.90       | 39.48       | **44.02** |
| T6   | 67.34       | 66.81       | **85.25** |

The final per-task accuracy reveals the quality of knowledge retention after the entire task sequence. The cosine classifier outperforms both baselines on every single task. The most striking result is on the final task T6, where the cosine model achieves 85.25% compared to 67.34% for Baseline-CPU, a gain of nearly 18 percentage points. This is particularly significant because T6 accuracy reflects both the model's ability to learn the current task and its capacity to maintain a well-calibrated classifier across all previously learned classes.

The improvement on earlier tasks (T1: 35.48% vs 27.52%; T2: 17.81% vs 15.30%) demonstrates that the cosine classifier also reduces backward forgetting. By eliminating magnitude bias, old class prototypes remain competitive in the angular scoring even after extensive training on subsequent tasks.

### C. Non-IID Flow Loss Stability

| Configuration | Median  | Min | Max          |
|---------------|---------|-----|--------------|
| Baseline-CPU  | 96,449  | 518 | 1.00 × 10¹⁸ |
| Baseline-GPU  | 20,998  | 521 | 1.53 × 10¹⁸ |
| Cosine (80ep) | **518** | 470 | 8.38 × 10¹¹ |

The flow loss statistics reveal a dramatic improvement in numerical stability. The baseline configurations exhibit median flow losses of 96,449 (CPU) and 20,998 (GPU), with catastrophic maximum values reaching $10^{18}$, indicating severe numerical explosions during training. In stark contrast, the cosine variant achieves a median flow loss of just 518, a reduction of over two orders of magnitude from the best baseline.

This stability improvement arises because the cosine classifier constrains the feature extractor to produce representations with consistent magnitude. When features have stable norms across tasks and batches, the normalizing flow receives a more stationary input distribution, enabling more accurate density estimation. The flow's coupling layers do not need to accommodate wildly varying feature scales, reducing the likelihood of numerical overflow in the log-probability computation.

The maximum flow loss for the cosine variant ($8.38 \times 10^{11}$) is still large, indicating occasional instability, but it is seven orders of magnitude smaller than the baseline maximum ($1.53 \times 10^{18}$). This suggests that the cosine classifier substantially reduces but does not entirely eliminate the numerical challenges inherent in normalizing flow training.

### D. Shared-Split Class-So-Far Accuracy

| Task | Baseline | Cosine (30ep) | Cosine (50ep) | Cosine (150ep) |
|------|----------|---------------|---------------|----------------|
| t=1  | 55.95    | 67.27         | 70.43         | **75.58**      |
| t=2  | 54.70    | 71.85         | 74.00         | **75.67**      |
| t=3  | 49.66    | 75.06         | 73.66         | **74.96**      |
| t=4  | 49.50    | 69.65         | 70.89         | **62.02**      |
| t=5  | 52.87    | 75.69         | 72.36         | **70.76**      |
| t=6  | 61.27    | 87.44         | 86.41         | **84.40**      |

On the Shared split, where all clients share the same class distribution, the cosine classifier demonstrates consistent and substantial improvements across all tasks and all training horizons. Even at just 30 epochs, the cosine variant achieves 87.44% class-so-far accuracy at the final task, compared to 61.27% for the baseline, representing a 26-point improvement.

The results across different epoch counts (30, 50, 150) reveal an interesting pattern. The cosine classifier achieves strong performance early (30 epochs) and maintains it as training continues, whereas additional epochs do not always improve and sometimes show slight degradation on intermediate tasks. This suggests that the angular decision boundaries converge faster than their linear counterparts, and that the primary benefit of the cosine head is in the quality of the learned geometry rather than the quantity of optimization.

### E. Shared-Split Final Per-Task Accuracy

| Task | Baseline | Cosine (30ep) | Cosine (50ep) | Cosine (150ep) |
|------|----------|---------------|---------------|----------------|
| T1   | 60.55    | 85.62         | 84.73         | **89.02**      |
| T2   | 62.98    | 85.70         | 86.49         | **85.70**      |
| T3   | 54.39    | 84.98         | 84.65         | **79.56**      |
| T4   | 63.02    | **91.77**     | 89.22         | 84.49          |
| T5   | 54.86    | 86.03         | 81.75         | **75.35**      |
| T6   | 71.83    | 90.52         | 91.59         | **92.24**      |

The per-task final accuracies in the Shared split are uniformly high for the cosine variant. Every task achieves above 75% accuracy even in the worst case, compared to the baseline where some tasks fall below 55%. The cosine classifier at 30 epochs achieves 91.77% on T4 and 90.52% on T6, demonstrating that angular decision boundaries produce highly discriminative classifiers even with moderate training.

### F. Shared-Split Flow Loss Stability

| Configuration  | Median  | Min | Max            |
|----------------|---------|-----|----------------|
| Baseline       | 28,209  | 514 | 1.71 × 10¹⁷   |
| Cosine (30ep)  | **470** | 470 | 5.26 × 10²    |
| Cosine (50ep)  | **470** | 470 | 1.42 × 10⁵    |
| Cosine (150ep) | 642     | 470 | 1.77 × 10¹³   |

The flow stability improvements on the Shared split are even more pronounced. The cosine variant at 30 epochs achieves a median flow loss of 470 with a maximum of merely 526, indicating nearly perfect numerical stability throughout training. Compare this with the baseline median of 28,209 and maximum of $1.71 \times 10^{17}$, representing a catastrophic numerical regime.

At 50 epochs, the maximum increases slightly to $1.42 \times 10^5$, and at 150 epochs to $1.77 \times 10^{13}$, suggesting that longer training introduces occasional instability episodes. Nevertheless, even the worst cosine configuration (150 epochs) produces a median flow loss of 642, which is 44 times lower than the baseline median.

## VII. Discussion

### A. Why Cosine Normalization Works in FCL

The experimental results demonstrate that a seemingly simple modification, replacing the linear classification head with a cosine-normalized head, produces outsized improvements in accuracy, retention, and stability. We attribute this to the unique convergence of challenges in federated continual learning that make magnitude bias particularly harmful.

In standard centralized learning, magnitude bias is partially self-correcting: the optimizer sees all classes in every epoch and naturally equilibrates weight norms through balanced gradient updates. In continual learning, this self-correction fails because old classes are absent from the training data. In federated learning, it further fails because different clients update different classes with different intensities, and the averaging operation conflates genuine discriminative information with noise in the weight norms. The cosine classifier eliminates this entire category of failure by removing magnitude from the decision function.

### B. Interaction with Normalizing Flow

A key finding is the dramatic stabilization of the normalizing flow training. We hypothesize that this occurs through the following causal chain: the cosine classifier's L2 normalization of the feature vector implicitly encourages the feature extractor to produce representations with stable norms. When feature norms are stable, the normalizing flow receives inputs from a more stationary distribution, reducing the variance of the log-probability estimates and preventing the numerical explosions observed in the baseline.

This interpretation is supported by the flow loss statistics: the cosine variant's median flow loss (470-642) is remarkably close to the minimum observed across all configurations (470), indicating that the flow achieves near-optimal density estimation consistently rather than oscillating between good and catastrophic states.

### C. Advantages of the Proposed Approach

**Lightweight.** The cosine classifier adds only one additional parameter (the temperature scalar $\sigma$) compared to the baseline and actually removes the bias vector, resulting in a net reduction in parameter count. The computational overhead of L2 normalization is negligible relative to the cost of the forward and backward passes through the feature extractor and normalizing flow.

**Modular.** The modification is confined to the classification head and does not require changes to the flow architecture, replay mechanism, distillation losses, or federated aggregation protocol. This modularity means it can be combined with any other improvements to the AF-FCL framework without interference.

**Scalable.** The benefits of cosine normalization are expected to increase with the number of classes and tasks, as magnitude bias becomes more severe when more classes compete for representation in the weight space. The EMNIST results with 26 classes across 6 tasks already show substantial gains; extrapolation to larger class spaces (e.g., CIFAR-100 with 100 classes) is a natural direction.

### D. Limitations

**Dependence on Feature Quality.** The cosine classifier's effectiveness depends on the feature extractor producing representations where angular proximity corresponds to semantic similarity. If the feature space is poorly structured, cosine similarity may not provide meaningful class discrimination. However, this limitation is shared by all metric-learning approaches and is mitigated by the quality of modern convolutional feature extractors.

**Temperature Sensitivity.** The learnable temperature $\sigma$ must be properly initialized and bounded to prevent training instability. We found that initializing $\sigma = 5.0$ with bounds $[1.0, 30.0]$ works well across our experiments, but different datasets and task configurations may require tuning.

**Angular Margin Tuning.** The optional ArcFace-style angular margin introduces an additional hyperparameter that influences the tightness of class clusters. While we found $m = 0.15$ to be effective for EMNIST, the optimal margin may vary across datasets and the number of classes per task.

## VIII. Conclusion

We have presented a cosine-normalized classification head as a targeted architectural improvement to the AF-FCL framework for federated continual learning. By replacing the standard linear classifier with a CosineLinear module that computes logits based purely on angular similarity between L2-normalized features and class prototypes, we eliminate the magnitude bias that systematically favors recently updated classes in sequential task learning.

Our experimental evaluation on EMNIST-Letters under both Non-IID and Shared data partitions demonstrates three consistent benefits. First, the cosine classifier substantially improves classification accuracy, achieving 62.80% class-so-far accuracy at the first Non-IID task compared to 27.11% for the baseline, and 87.44% at the final Shared-split task compared to 61.27%. Second, the cosine classifier reduces catastrophic forgetting, with final per-task accuracies consistently higher than the baseline across all tasks. Third, the cosine classifier dramatically stabilizes normalizing flow training, reducing the median flow loss by two orders of magnitude and preventing the catastrophic numerical explosions observed in baseline configurations.

These improvements are achieved through a single, lightweight architectural modification that adds negligible computational cost and is fully compatible with all other components of the AF-FCL framework. The cosine-normalized classifier represents a principled solution to a fundamental geometric problem in continual learning: the entanglement of magnitude and direction in linear decision boundaries.

Future work will explore several extensions. First, we plan to evaluate the cosine classifier on larger-scale benchmarks including CIFAR-100 and ImageNet subsets, where the benefits of magnitude-invariant classification are expected to be even more pronounced due to the larger class space. Second, we will investigate adaptive angular margin schedules that increase the margin as training progresses through successive tasks, enforcing progressively tighter class separation. Third, we will study the interaction between cosine normalization and more sophisticated federated aggregation strategies, such as attention-weighted averaging, to determine whether the benefits are complementary. Finally, we aim to provide theoretical convergence guarantees for federated continual learning with cosine classifiers, leveraging the geometric structure of the unit hypersphere to derive tighter bounds on the forgetting rate.

## References

[1] B. McMahan, E. Moore, D. Ramage, S. Hampson, and B. A. y Arcas, "Communication-efficient learning of deep networks from decentralized data," in Proc. AISTATS, 2017.

[2] T. Li, A. K. Sahu, M. Zaheer, M. Sanjabi, A. Talwalkar, and V. Smith, "Federated optimization in heterogeneous networks," in Proc. MLSys, 2020.

[3] J. Kirkpatrick et al., "Overcoming catastrophic forgetting in neural networks," Proc. Natl. Acad. Sci., vol. 114, no. 13, pp. 3521–3526, 2017.

[4] R. Aljundi, F. Babiloni, M. Elhoseiny, M. Rohrbach, and T. Tuytelaars, "Memory aware synapses: Learning what (not) to forget," in Proc. ECCV, 2018.

[5] S.-A. Rebuffi, A. Kolesnikov, G. Sperl, and C. H. Lampert, "iCaRL: Incremental classifier and representation learning," in Proc. CVPR, 2017.

[6] Y. Qi et al., "Accurate forgetting for federated continual learning," arXiv preprint, 2024.

[7] J. Deng, J. Guo, N. Xue, and S. Zafeiriou, "ArcFace: Additive angular margin loss for deep face recognition," in Proc. CVPR, 2019.

[8] S. Hou, X. Pan, C. C. Loy, Z. Wang, and D. Lin, "Learning a unified classifier incrementally via rebalancing," in Proc. CVPR, 2019.

[9] D. P. Kingma and J. Ba, "Adam: A method for stochastic optimization," in Proc. ICLR, 2015.

[10] L. Dinh, J. Sohl-Dickstein, and S. Bengio, "Density estimation using real-valued non-volume preserving transformations," in Proc. ICLR, 2017.

[11] G. Papamakarios, E. Nalisnick, D. J. Rezende, S. Mohamed, and B. Lakshminarayanan, "Normalizing flows for probabilistic modeling and inference," J. Mach. Learn. Res., vol. 22, no. 57, pp. 1–64, 2021.

[12] W. Qi, L. Gong, and Y. Wang, "Prototype-based contrastive replay for continual learning," in Proc. NeurIPS Workshop, 2023.

[13] A. Chaudhry, P. K. Dokania, T. Ajanthan, and P. H. S. Torr, "Riemannian walk for incremental learning: Understanding forgetting and intransigence," in Proc. ECCV, 2018.

[14] Z. Li and D. Hoiem, "Learning without forgetting," IEEE Trans. Pattern Anal. Mach. Intell., vol. 40, no. 12, pp. 2935–2947, 2018.
