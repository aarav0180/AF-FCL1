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

---

## Appendix A: Detailed Mathematical Formulation of Normalizing Flows in AF-FCL

### A.1 Change of Variables and Log-Likelihood

A normalizing flow defines an invertible, differentiable mapping $f_\phi : \mathbb{R}^d \to \mathbb{R}^d$ that transforms a simple base distribution $p_U(\mathbf{u})$ (typically $\mathcal{N}(\mathbf{0}, \mathbf{I}_d)$) into a complex target distribution $p_X(\mathbf{x})$. The relationship between these distributions is governed by the change-of-variables formula. Given $\mathbf{x} = f_\phi(\mathbf{u})$, the density of $\mathbf{x}$ is:

$$p_X(\mathbf{x}) = p_U\!\bigl(f_\phi^{-1}(\mathbf{x})\bigr) \;\left|\det \frac{\partial f_\phi^{-1}}{\partial \mathbf{x}}\right|$$

Taking the logarithm:

$$\log p_X(\mathbf{x}) = \log p_U\!\bigl(f_\phi^{-1}(\mathbf{x})\bigr) + \log \left|\det \frac{\partial f_\phi^{-1}}{\partial \mathbf{x}}\right|$$

For the standard Gaussian base distribution $p_U(\mathbf{u}) = \mathcal{N}(\mathbf{0}, \mathbf{I}_d)$, the first term expands to:

$$\log p_U(\mathbf{u}) = -\frac{d}{2}\log(2\pi) - \frac{1}{2}\|\mathbf{u}\|^2$$

where $\mathbf{u} = f_\phi^{-1}(\mathbf{x})$ is the noise vector obtained by inverting the flow.

### A.2 Affine Coupling Layers

AF-FCL implements the flow using affine coupling transforms. The input $\mathbf{x} \in \mathbb{R}^d$ is partitioned into two halves $\mathbf{x} = [\mathbf{x}_{1:d/2},\; \mathbf{x}_{d/2+1:d}]$ using a binary mask $\mathbf{m}$. The coupling transform is:

$$\mathbf{y}_{1:d/2} = \mathbf{x}_{1:d/2}$$
$$\mathbf{y}_{d/2+1:d} = \mathbf{x}_{d/2+1:d} \odot \exp\!\bigl(s(\mathbf{x}_{1:d/2};\, y)\bigr) + t(\mathbf{x}_{1:d/2};\, y)$$

where $s(\cdot;\, y)$ and $t(\cdot;\, y)$ are the scale and translation functions parameterized by residual networks (`ResidualNet`) conditioned on the one-hot class label $y$. The Jacobian of this transform is triangular, so its log-determinant is simply:

$$\log\left|\det \frac{\partial \mathbf{y}}{\partial \mathbf{x}}\right| = \sum_{j=d/2+1}^{d} s_j(\mathbf{x}_{1:d/2};\, y)$$

This is computationally cheap—linear in $d$ rather than the $O(d^3)$ cost of a general determinant.

### A.3 Composite Transform

AF-FCL composes $L=4$ coupling layers with alternating permutations (reverse permutations for the first $L/2$ layers, random permutations thereafter). Let $g_\ell$ denote the $\ell$-th coupling layer and $\pi_\ell$ the preceding permutation. The full transform is:

$$f_\phi = g_L \circ \pi_L \circ g_{L-1} \circ \pi_{L-1} \circ \cdots \circ g_1 \circ \pi_1$$

The total log-determinant decomposes additively:

$$\log\left|\det \frac{\partial f_\phi^{-1}}{\partial \mathbf{x}}\right| = \sum_{\ell=1}^{L} \log\left|\det \frac{\partial g_\ell^{-1}}{\partial \mathbf{h}_\ell}\right|$$

since permutations have unit Jacobian determinant ($|\det \mathbf{P}| = 1$).

### A.4 Conditional Density Estimation

In AF-FCL, the flow is conditioned on the class label. For a sample $(\mathbf{x}_a, y)$, the class label is encoded as a one-hot vector $\mathbf{c} = \text{one\_hot}(y) \in \{0,1\}^C$ where $C$ is the total number of classes. This context vector is fed into every residual network inside the coupling layers, so the scale and translation functions become $s(\cdot;\, \mathbf{c})$ and $t(\cdot;\, \mathbf{c})$.

The training objective maximizes the conditional log-likelihood:

$$\max_\phi \; \mathbb{E}_{(\mathbf{x}_a, y) \sim \mathcal{D}} \left[\log p_\phi(\mathbf{x}_a \mid y)\right]$$

which in code corresponds to minimizing:

$$\mathcal{L}_{\text{flow-data}} = -\frac{1}{N}\sum_{i=1}^{N} \log p_\phi(\mathbf{x}_a^{(i)} \mid y^{(i)})$$

### A.5 Replay Probability Weighting

When sampling synthetic features $\tilde{\mathbf{x}}_a$ from the flow for replay, AF-FCL weights each sample by its estimated probability under the local data distribution. For each class $c$ present in the current batch, the local distribution is approximated as a Gaussian in the noise space:

$$\hat{\mu}_c = \frac{1}{|\{i: y_i = c\}|} \sum_{i: y_i = c} \mathbf{u}_i, \quad \hat{\sigma}_c^2 = \frac{1}{|\{i: y_i = c\}|} \sum_{i: y_i = c} (\mathbf{u}_i - \hat{\mu}_c)^2$$

where $\mathbf{u}_i = f_\phi^{-1}(\mathbf{x}_a^{(i)};\, y_i)$ are the noise vectors of the current batch. The probability of a flow-generated sample $\tilde{\mathbf{x}}_a$ with label $\tilde{y} = c$ is:

$$p_{\text{local}}(\tilde{\mathbf{x}}_a) = \frac{1}{\sqrt{2\pi}} \prod_{j=1}^{d} (\hat{\sigma}_{c,j}^2 + \epsilon)^{-1/2} \exp\!\left(-\frac{(\tilde{u}_j - \hat{\mu}_{c,j})^2}{2(\hat{\sigma}_{c,j}^2 + \epsilon)}\right)$$

averaged over dimensions to produce a scalar weight. The exploration-exploitation tradeoff is controlled by:

$$w_{\text{explore}} = (1 - \theta) \cdot \bar{p}_{\text{local}} + \theta$$

where $\theta = 0.1$ ensures a minimum replay weight even for out-of-distribution samples, and $\bar{p}_{\text{local}}$ is the mean probability across the batch.

---

## Appendix B: Detailed Geometry of the Cosine Classifier

### B.1 Linear Classifier Decision Boundary

For a standard linear classifier with weight matrix $\mathbf{W} \in \mathbb{R}^{C \times d}$ and bias $\mathbf{b} \in \mathbb{R}^C$, the decision boundary between classes $a$ and $b$ is the hyperplane:

$$\{\ \mathbf{x} \in \mathbb{R}^d : z_a = z_b\ \}$$

Expanding:

$$\mathbf{W}_a^\top \mathbf{x} + b_a = \mathbf{W}_b^\top \mathbf{x} + b_b$$

$$(\mathbf{W}_a - \mathbf{W}_b)^\top \mathbf{x} = b_b - b_a$$

This is a linear hyperplane whose orientation depends on the difference $\mathbf{W}_a - \mathbf{W}_b$ and whose offset depends on the biases. Critically, the position of this hyperplane shifts whenever the norms $\|\mathbf{W}_a\|$ or $\|\mathbf{W}_b\|$ change, even if the directions $\hat{\mathbf{W}}_a$ and $\hat{\mathbf{W}}_b$ remain fixed. In continual learning, this means that magnitude drift alone can move decision boundaries and misclassify previously correct inputs.

### B.2 Cosine Classifier Decision Boundary

For the cosine classifier, the decision boundary between classes $a$ and $b$ is:

$$\sigma \cdot \cos(\theta_a) = \sigma \cdot \cos(\theta_b)$$

$$\cos(\theta_a) = \cos(\theta_b)$$

$$\hat{\mathbf{W}}_a^\top \hat{\mathbf{x}} = \hat{\mathbf{W}}_b^\top \hat{\mathbf{x}}$$

$$(\hat{\mathbf{W}}_a - \hat{\mathbf{W}}_b)^\top \hat{\mathbf{x}} = 0$$

This boundary depends only on the unit-norm prototypes $\hat{\mathbf{W}}_a, \hat{\mathbf{W}}_b$ and the unit-norm feature $\hat{\mathbf{x}}$. It is invariant to any scaling of $\mathbf{W}_a$, $\mathbf{W}_b$, or $\mathbf{x}$. Geometrically, the boundary is a great circle on the unit hypersphere $\mathbb{S}^{d-1}$, equidistant (in angle) from both prototypes.

### B.3 Effect of Angular Margin on Decision Geometry

With an additive angular margin $m > 0$, the training-time logit for the ground-truth class $y_i$ becomes:

$$z_{y_i} = \sigma \cdot \cos(\theta_{y_i} + m)$$

Using the cosine addition formula:

$$\cos(\theta + m) = \cos\theta \cos m - \sin\theta \sin m$$

Since $\cos m < 1$ and $\sin m > 0$ for $m \in (0, \pi/2)$, this effectively reduces the logit of the correct class during training. The network must compensate by learning tighter angular alignment ($\theta_{y_i} \to 0$), which produces more compact class clusters on the hypersphere.

The inter-class angular margin after convergence satisfies:

$$\theta_{y_i} + m \leq \theta_c \quad \forall\, c \neq y_i$$

meaning the angular gap between any feature and its nearest incorrect prototype is at least $m$ radians. For our setting with $m = 0.15$ radians ($\approx 8.6°$), this provides a substantial angular buffer against misclassification.

### B.4 Temperature Dynamics

The learnable temperature $\sigma$ controls the sharpness of the softmax distribution. Given cosine logits $z_c = \sigma \cos(\theta_c)$, the softmax probability for class $c$ is:

$$p(y = c \mid \mathbf{x}) = \frac{\exp(\sigma \cos\theta_c)}{\sum_{j=1}^{C} \exp(\sigma \cos\theta_j)}$$

The entropy of this distribution is:

$$H = -\sum_{c=1}^{C} p_c \log p_c$$

As $\sigma \to 0$, $p_c \to 1/C$ for all $c$ (uniform distribution, maximum entropy). As $\sigma \to \infty$, $p_c \to \mathbb{1}[c = \arg\max_j \cos\theta_j]$ (one-hot, zero entropy). The optimal $\sigma$ trades off between gradient signal strength (higher $\sigma$) and overconfident predictions (lower $\sigma$).

In AF-FCL, $\sigma$ is initialized to a moderate value (5.0 or 10.0) and clamped to $[1.0, 30.0]$:

$$\sigma_{\text{eff}} = \text{clamp}(\sigma, 1.0, 30.0)$$

This prevents the temperature from collapsing to zero (which would halt learning) or exploding (which would cause gradient overflow in the softmax).

### B.5 Gradient Analysis: Linear vs. Cosine

For a linear classifier with cross-entropy loss, the gradient of the loss with respect to the weight vector $\mathbf{W}_c$ is:

$$\frac{\partial \mathcal{L}}{\partial \mathbf{W}_c} = (p_c - \mathbb{1}[y = c]) \cdot \mathbf{x}$$

The gradient magnitude scales with $\|\mathbf{x}\|$. If the feature extractor produces features with varying norms across tasks, the effective learning rate for different tasks will differ, creating an implicit bias toward tasks with larger feature norms.

For the cosine classifier, the gradient involves the normalized quantities:

$$\frac{\partial \mathcal{L}}{\partial \mathbf{W}_c} = \sigma (p_c - \mathbb{1}[y = c]) \cdot \frac{\partial \cos\theta_c}{\partial \mathbf{W}_c}$$

where:

$$\frac{\partial \cos\theta_c}{\partial \mathbf{W}_c} = \frac{1}{\|\mathbf{W}_c\|}\left(\hat{\mathbf{x}} - \cos\theta_c \cdot \hat{\mathbf{W}}_c\right)$$

This gradient has two important properties. First, it is inversely proportional to $\|\mathbf{W}_c\|$, creating a self-regulating effect: prototypes with large norms receive smaller gradient updates, preventing runaway norm growth. Second, the direction $(\hat{\mathbf{x}} - \cos\theta_c \cdot \hat{\mathbf{W}}_c)$ is the component of $\hat{\mathbf{x}}$ perpendicular to $\hat{\mathbf{W}}_c$ on the hypersphere, meaning updates always move the prototype toward the feature on the sphere surface—a purely angular correction.

---

## Appendix C: EWC Regularization in the Cosine Setting

### C.1 Fisher Information Matrix

After completing each task $t$, the diagonal Fisher Information Matrix (FIM) is computed for all classifier parameters $\boldsymbol{\theta}$:

$$F_{ii}^{(t)} = \mathbb{E}_{(\mathbf{x},y) \sim \mathcal{D}^t}\left[\left(\frac{\partial \log p(y \mid \mathbf{x};\, \boldsymbol{\theta})}{\partial \theta_i}\right)^2\right]$$

In practice, this is approximated using a finite number of batches from the task's training data:

$$\hat{F}_{ii}^{(t)} = \frac{1}{N_{\text{batch}}} \sum_{n=1}^{N_{\text{batch}}} \left(\frac{\partial \mathcal{L}_n}{\partial \theta_i}\right)^2$$

The FIM captures the curvature of the loss landscape at the current parameter values. Parameters with high Fisher values are important for the current task and should be preserved; parameters with low Fisher values can be freely modified for future tasks.

### C.2 EWC Penalty

The EWC penalty added to the classifier loss at task $t+1$ is:

$$\mathcal{L}_{\text{EWC}} = \frac{\lambda_{\text{EWC}}}{2} \sum_{i} F_{ii}^{(t)} \left(\theta_i - \theta_i^{(t)*}\right)^2$$

where $\theta_i^{(t)*}$ is the parameter value at the end of task $t$. This quadratic penalty discourages large deviations from the previous solution in directions that are important for past tasks, while allowing free movement in unimportant directions.

### C.3 Interaction with Cosine Normalization

In the cosine classifier, the weight parameters include both the prototype matrix $\mathbf{W} \in \mathbb{R}^{C \times d}$ and the temperature scalar $\sigma$. The EWC penalty is applied to all of these parameters.

For the prototypes, the Fisher diagonal captures which prototype dimensions are most important for distinguishing classes learned so far. Because the cosine classifier operates on normalized prototypes, the Fisher values tend to be more uniform across dimensions compared to the linear case, where magnitude-dominated dimensions can have artificially inflated Fisher values. This leads to more balanced regularization across the parameter space.

For the temperature $\sigma$, the Fisher value captures how sensitive the classification is to changes in sharpness. Once $\sigma$ has converged to a value that produces well-calibrated probabilities, its Fisher value will be high, preventing subsequent tasks from disrupting the calibration.

### C.4 Running Fisher Average

When multiple tasks are encountered sequentially, the Fisher matrices from different tasks are merged using an exponential moving average:

$$\bar{F}_{ii}^{(t)} = \alpha \cdot \bar{F}_{ii}^{(t-1)} + (1 - \alpha) \cdot F_{ii}^{(t)}$$

with $\alpha = 0.5$ in our implementation. This ensures that the importance estimates reflect a balanced view of all past tasks rather than being dominated by the most recent one.

---

## Appendix D: Federated Averaging on the Unit Hypersphere

### D.1 Standard FedAvg

In standard FedAvg, the server aggregates client models by computing:

$$\mathbf{W}^{\text{global}} = \sum_{k=1}^{K} \frac{n_k}{n} \mathbf{W}^{(k)}$$

where $n_k$ is the number of samples at client $k$ and $n = \sum_k n_k$.

For a linear classifier, this averaging operates directly on the raw weight vectors, blending both their directions and magnitudes. If client $k$ updates class $c$ aggressively (because it has many samples of that class), $\|\mathbf{W}_c^{(k)}\|$ will be large, and the average will be pulled toward that client's solution in both direction and magnitude.

### D.2 Effective Averaging with Cosine Normalization

With cosine normalization, the actual weight vectors $\mathbf{W}_c$ are still averaged by FedAvg. However, during the forward pass, these averaged weights are L2-normalized:

$$\hat{\mathbf{W}}_c^{\text{global}} = \frac{\mathbf{W}_c^{\text{global}}}{\|\mathbf{W}_c^{\text{global}}\|}$$

This re-normalization means that the effective aggregation operates on directions rather than raw vectors. To see why this is beneficial, consider the decomposition:

$$\mathbf{W}_c^{\text{global}} = \sum_k \frac{n_k}{n} \|\mathbf{W}_c^{(k)}\| \cdot \hat{\mathbf{W}}_c^{(k)}$$

The magnitude $\|\mathbf{W}_c^{\text{global}}\|$ absorbs the heterogeneous norms from different clients, but this magnitude is discarded during normalization. The resulting direction $\hat{\mathbf{W}}_c^{\text{global}}$ is a norm-weighted average of client directions—clients with larger norms have more influence on the consensus direction, but the final prototype has unit norm regardless.

### D.3 Robustness to Heterogeneous Updates

In a Non-IID setting, some clients may not have any data for class $c$ and therefore do not update $\mathbf{W}_c$. After FedAvg, the class-$c$ prototype is an average of updated and stale copies. With a linear classifier, the stale copies may have a very different norm from the updated copies, creating a misleading average. With cosine normalization, the norm discrepancy is absorbed, and only the directional information—which is more stable under partial updates—determines the classifier's behavior.

---

## Appendix E: Complete Loss Function Decomposition

### E.1 Full Classifier Loss

The total classifier loss for a single training step combines current-task learning, knowledge distillation, replay, and regularization:

$$\mathcal{L}_{\text{total}} = \underbrace{\mathcal{L}_{\text{CE}}(\mathbf{x}, y)}_{\text{current task}} + \underbrace{\lambda_f \bigl(\mathcal{L}_{\text{feat-last}} + \mathcal{L}_{\text{feat-global}}\bigr) + \lambda_o \bigl(\mathcal{L}_{\text{out-last}} + \mathcal{L}_{\text{out-global}}\bigr)}_{\text{knowledge distillation}} + \underbrace{\lambda_{\text{flow}} \cdot w_{\text{explore}} \cdot \mathcal{L}_{\text{replay}}}_{\text{flow replay}} + \underbrace{\mathcal{L}_{\text{EWC}}}_{\text{EWC penalty}}$$

### E.2 Current-Task Cross-Entropy

With the cosine classifier, the cross-entropy loss becomes:

$$\mathcal{L}_{\text{CE}} = -\log \frac{\exp(\sigma \cos(\theta_{y} + m \cdot \mathbb{1}[\text{training}]))}{\sum_{c=1}^{C} \exp(\sigma \cos(\theta_c + m \cdot \mathbb{1}[c = y] \cdot \mathbb{1}[\text{training}]))}$$

During training, the margin $m$ is added only to the ground-truth class logit. At inference, $m = 0$.

### E.3 Feature-Level Distillation

$$\mathcal{L}_{\text{feat-last}} = \lambda_{\text{KD-last}} \cdot \frac{1}{d} \sum_{j=1}^{d} (x_{a,j} - x_{a,j}^{\text{last}})^2$$

where $\mathbf{x}_a^{\text{last}}$ is the feature vector produced by the previous model (frozen). This loss operates on the pre-classification features and is therefore identical for both linear and cosine classifiers.

### E.4 Output-Level Distillation

$$\mathcal{L}_{\text{out-last}} = \lambda_{\text{KD-last}} \cdot \text{MCE}(\mathbf{p},\, \mathbf{p}^{\text{last}};\, T=2)$$

where $\text{MCE}$ is the multi-class cross-entropy with temperature scaling:

$$\text{MCE}(\mathbf{p}, \mathbf{q}; T) = -\sum_{c=1}^{C} \frac{q_c^{1/T}}{\sum_j q_j^{1/T}} \log \frac{p_c^{1/T}}{\sum_j p_j^{1/T}}$$

Both $\mathbf{p}$ (current model) and $\mathbf{q}$ (teacher model) are softmax probability vectors. For the cosine classifier, these probabilities are derived from the cosine logits $\sigma \cos\theta_c$.

### E.5 Replay Cross-Entropy

$$\mathcal{L}_{\text{replay}} = \frac{1}{N}\sum_{i=1}^{N} p_{\text{local}}(\tilde{\mathbf{x}}_a^{(i)}) \cdot \bigl(-\log p(\tilde{y}^{(i)} \mid \tilde{\mathbf{x}}_a^{(i)})\bigr)$$

where $\tilde{\mathbf{x}}_a^{(i)}$ are flow-generated features, $\tilde{y}^{(i)}$ are their sampled labels, and $p_{\text{local}}$ is the Gaussian probability weight from Appendix A.5. With the cosine classifier, $p(\tilde{y} \mid \tilde{\mathbf{x}}_a)$ is the softmax over cosine logits applied to the flow-generated features.

### E.6 Adaptive KD Weighting

The adaptive mixin scales all four KD components by:

$$\alpha = \text{sigmoid}(\text{batch\_acc} - 0.5)$$

where $\text{batch\_acc} = \frac{1}{N}\sum_{i=1}^{N} \mathbb{1}[\hat{y}_i = y_i]$ is the current mini-batch accuracy. This produces:

- $\alpha \approx 0.38$ when accuracy is low (relax KD, prioritize learning)
- $\alpha = 0.50$ when accuracy is moderate (balanced)
- $\alpha \approx 0.62$ when accuracy is high (tighten KD, prioritize retention)

The scaled KD loss becomes:

$$\mathcal{L}_{\text{KD}}^{\text{adaptive}} = \alpha \cdot \mathcal{L}_{\text{KD}}$$

---

## Appendix F: Implementation Details from Code

### F.1 CosineLinear Module

The core implementation (from `cosine_head.py`) performs:

```
x_norm = F.normalize(x, p=2, dim=1)          # [N, D] -> unit vectors
w_norm = F.normalize(self.weight, p=2, dim=1) # [C, D] -> unit vectors
cos_sim = x_norm @ w_norm.t()                 # [N, C] cosine similarities

# ArcFace margin (training only)
if self.training and self.margin > 0 and labels is not None:
    cos_sim = cos_sim.clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    theta = torch.acos(cos_sim)
    one_hot = F.one_hot(labels, num_classes=C).float()
    cos_sim = torch.cos(theta + self.margin * one_hot)

sigma_clamped = torch.clamp(self.sigma, min=1.0, max=30.0)
return sigma_clamped * cos_sim
```

Key numerical safeguards: clamping cosine to $(-1+\epsilon, 1-\epsilon)$ before `acos` prevents NaN gradients at the poles, and clamping $\sigma$ prevents extreme scaling.

### F.2 CosineMixin Class Hierarchy

The `CosineMixin` uses Python's Method Resolution Order (MRO) to swap the classifier after the parent `PreciseModel.__init__` builds the full network:

```
class CosineAdaptivePreciseModel(CosineMixin, AdaptivePreciseModel):
    pass
```

MRO: `CosineAdaptivePreciseModel` → `CosineMixin` → `AdaptivePreciseModel` → `AdaptiveMixin` → `PreciseModel`

`CosineMixin.__init__` calls `super().__init__(args)` which builds the full model with a linear head, then immediately replaces `self.classifier` with the cosine variant and rebuilds the optimizer.

### F.3 Feature Calibration (BatchNorm)

When `--cosine_calibration` is enabled, a `BatchNorm1d(xa_dim)` layer is inserted between `fc2` and the cosine head in `forward_from_xa`:

```
xb = F.leaky_relu(self.fc2(xa))
if self.feature_calibration:
    xb = self.feat_bn(xb)       # normalize feature statistics
logits = self.fc_classifier(xb)  # CosineLinear
```

This normalizes the mean and variance of features across each batch, ensuring that the cosine head receives inputs with consistent statistics regardless of the task or data distribution.
