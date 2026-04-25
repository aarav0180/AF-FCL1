from torch import nn
import torch
from torch import optim
import glog as logger
import numpy as np
import torch.nn.functional as F

from FLAlgorithms.PreciseFCLNet.classify_net import S_ConvNet, Resnet_plus
from nflows.flows.base import Flow
from nflows.transforms.permutations import RandomPermutation, ReversePermutation
from nflows.transforms.base import CompositeTransform
from nflows.transforms.coupling import AffineCouplingTransform
from nflows.nn.nets.myresnet import ResidualNet
from torch.nn import functional as F
from nflows.distributions.normal import StandardNormal
from utils.utils import myitem

eps = 1e-30


class FeaturePermutation(nn.Module):
    def __init__(self, feature_dim, mode='identity', seed=0):
        super().__init__()
        if mode == 'reverse':
            perm = torch.arange(feature_dim - 1, -1, -1)
        elif mode == 'random':
            generator = torch.Generator()
            generator.manual_seed(seed)
            perm = torch.randperm(feature_dim, generator=generator)
        else:
            perm = torch.arange(feature_dim)

        inv_perm = torch.empty_like(perm)
        inv_perm[perm] = torch.arange(feature_dim)

        self.register_buffer('perm', perm.long())
        self.register_buffer('inv_perm', inv_perm.long())

    def forward(self, x):
        return x[..., self.perm]

    def inverse(self, x):
        return x[..., self.inv_perm]


class MultiScaleFlow(nn.Module):
    def __init__(self, flows, permutations, temperature=1.0):
        super().__init__()
        self.flows = nn.ModuleList(flows)
        self.permutations = nn.ModuleList(permutations)
        self.scale_logits = nn.Parameter(torch.zeros(len(flows)))
        self.temperature = temperature

    def log_prob(self, inputs, context=None):
        weights = torch.softmax(self.scale_logits / max(self.temperature, 1e-6), dim=0)
        log_probs = []
        for flow, permutation in zip(self.flows, self.permutations):
            transformed_inputs = permutation(inputs)
            log_probs.append(flow.log_prob(inputs=transformed_inputs, context=context))

        stacked = torch.stack(log_probs, dim=0)
        return torch.logsumexp(torch.log(weights + 1e-12).unsqueeze(1) + stacked, dim=0)

    def log_prob_and_noise(self, inputs, context=None):
        weights = torch.softmax(self.scale_logits / max(self.temperature, 1e-6), dim=0)
        branch = int(torch.argmax(weights).item())
        transformed_inputs = self.permutations[branch](inputs)
        log_prob, noise = self.flows[branch].log_prob_and_noise(transformed_inputs, context=context)
        return log_prob, self.permutations[branch].inverse(noise)

    def sample(self, num_samples, context=None):
        weights = torch.softmax(self.scale_logits / max(self.temperature, 1e-6), dim=0)
        branch = int(torch.argmax(weights).item())
        samples = self.flows[branch].sample(num_samples=num_samples, context=context)
        return self.permutations[branch].inverse(samples)

def MultiClassCrossEntropy(logits, labels, T):
    logits = torch.pow(logits+eps, 1/T)
    logits = logits/(torch.sum(logits, dim=1, keepdim=True)+eps)
    labels = torch.pow(labels+eps, 1/T)
    labels = labels/(torch.sum(labels, dim=1, keepdim=True)+eps)

    outputs = torch.log(logits+eps)
    outputs = torch.sum(outputs * labels, dim=1, keepdim=False)
    outputs = -torch.mean(outputs, dim=0, keepdim=False)
    return outputs

class PreciseModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        
        beta1 = args.beta1
        beta2 = args.beta2
        weight_decay = args.weight_decay
        lr = args.lr
        flow_lr = args.flow_lr
        c_channel_size = args.c_channel_size
        dataset = args.dataset

        self.algorithm = args.algorithm

        self.k_loss_flow = args.k_loss_flow
        self.k_kd_global_cls = args.k_kd_global_cls
        self.k_kd_last_cls = args.k_kd_last_cls
        self.k_kd_feature = args.k_kd_feature
        self.k_kd_output = args.k_kd_output
        self.k_flow_lastflow = args.k_flow_lastflow

        self.gcar = getattr(args, 'gcar', False)
        self.hmce = getattr(args, 'hmce', False)
        self.hmce_scales = max(int(getattr(args, 'hmce_scales', 3)), 1)
        self.cpr = getattr(args, 'cpr', False)
        self.cpr_tau = float(getattr(args, 'cpr_tau', 0.2))
        self.cpr_lambda = float(getattr(args, 'cpr_lambda', 0.1))
        self.maft = getattr(args, 'maft', False)
        self.maft_hidden = int(getattr(args, 'maft_hidden', 16))

        self.flow_explore_theta = args.flow_explore_theta
        self.fedprox_k = args.fedprox_k

        self.classify_criterion = nn.NLLLoss()
        self.classify_criterion_noreduce = nn.NLLLoss(reduction='none')

        self.flow = None
        if 'EMNIST-Letters' in dataset:
            # self.xa_shape=[128, 4, 4]
            self.xa_shape=[512]
            self.num_classes = 26
            self.classifier = S_ConvNet(28, 1, c_channel_size, xa_dim=int(np.prod(self.xa_shape)), num_classes=self.num_classes)
            if self.algorithm == 'PreciseFCL':
                if self.hmce:
                    self.flow = self.get_hmce_nflow_model(feature_dim=int(np.prod(self.xa_shape)), hidden_feature=512, context_feature=self.num_classes,
                                                          num_layers=4, num_scales=self.hmce_scales)
                else:
                    self.flow = self.get_1d_nflow_model(feature_dim=int(np.prod(self.xa_shape)), hidden_feature=512, context_feature=self.num_classes,
                                                    num_layers=4)
        elif dataset=='CIFAR100':
            self.xa_shape=[512]
            self.num_classes = 100
            self.classifier = Resnet_plus(32, xa_dim=int(np.prod(self.xa_shape)), num_classes=self.num_classes)
            # self.classifier = S_ConvNet(32, 3, c_channel_size, xa_dim=int(np.prod(self.xa_shape)), num_classes=self.num_classes)
            if self.algorithm == 'PreciseFCL':
                if self.hmce:
                    self.flow = self.get_hmce_nflow_model(feature_dim=int(np.prod(self.xa_shape)), hidden_feature=512, context_feature=self.num_classes,
                                                          num_layers=4, num_scales=self.hmce_scales)
                else:
                    self.flow = self.get_1d_nflow_model(feature_dim=int(np.prod(self.xa_shape)), hidden_feature=512, context_feature=self.num_classes,
                                                    num_layers=4)

        elif dataset=='MNIST-SVHN-FASHION':
            self.xa_shape=[512]
            self.num_classes = 20
            self.classifier = S_ConvNet(32, 3, c_channel_size, xa_dim=int(np.prod(self.xa_shape)), num_classes=self.num_classes)
            if self.algorithm == 'PreciseFCL':
                if self.hmce:
                    self.flow = self.get_hmce_nflow_model(feature_dim=int(np.prod(self.xa_shape)), hidden_feature=512, context_feature=self.num_classes,
                                                          num_layers=4, num_scales=self.hmce_scales)
                else:
                    self.flow = self.get_1d_nflow_model(feature_dim=int(np.prod(self.xa_shape)), hidden_feature=512, context_feature=self.num_classes,
                                                    num_layers=4)

        if self.maft:
            self.maft_gate = nn.Sequential(
                nn.Linear(3, self.maft_hidden),
                nn.ReLU(),
                nn.Linear(self.maft_hidden, 1),
                nn.Sigmoid(),
            )
        else:
            self.maft_gate = None

        classifier_params = list(self.classifier.parameters())
        if self.maft_gate is not None:
            classifier_params += list(self.maft_gate.parameters())

        self.classifier_optimizer = optim.Adam(
            classifier_params,
            lr=lr, weight_decay=weight_decay, betas=(beta1, beta2),
        )

        parameters_fb = [a[1] for a in filter(lambda x: 'fc2' in x[0], self.classifier.named_parameters())]
        self.classifier_fb_optimizer = optim.Adam(
            parameters_fb, lr=lr, weight_decay=weight_decay, 
            betas=(beta1, beta2),
        )

        if self.algorithm == 'PreciseFCL':
            self.flow_optimizer = optim.Adam(
                self.flow.parameters(), lr=flow_lr, 
                weight_decay=weight_decay, betas=(beta1, beta2),
            )

        class_params = sum(p.numel() for p in self.classifier.parameters())
        if self.algorithm == 'PreciseFCL':
            flow_params = sum(p.numel() for p in self.flow.parameters())
        else:
            flow_params = 0
        logger.info("Classifier model has %.3f M parameters; Flow model has %.3f M parameters", class_params / 1e6, flow_params / 1.0e6)

    def to(self, device):
        self.classifier.to(device)
        if self.maft_gate is not None:
            self.maft_gate.to(device)
        if self.algorithm == 'PreciseFCL':
            self.flow.to(device)
        self._device = device
        return self
    
    @property
    def device(self):
        return next(self.classifier.parameters()).device

    def parameters(self):
        for param in  self.classifier.parameters():
            yield param
        if self.maft_gate is not None:
            for param in self.maft_gate.parameters():
                yield param
        if self.algorithm == 'PreciseFCL':
            for param in  self.flow.parameters():
                yield param

    def named_parameters(self):
        for name, param in self.classifier.named_parameters():
            yield 'classifier.'+name, param
        if self.maft_gate is not None:
            for name, param in self.maft_gate.named_parameters():
                yield 'maft_gate.'+name, param
        if self.algorithm == 'PreciseFCL':
            for name, param in self.flow.named_parameters():
                yield 'flow.'+name, param

    def get_1d_nflow_model(self,
                        feature_dim, 
                        hidden_feature, 
                        context_feature,
                        num_layers):
        transforms = []
        
        for l in range(num_layers):
            assert num_layers//2>1
            if l < num_layers//2:
                transforms.append(ReversePermutation(features=feature_dim))
            else:
                transforms.append(RandomPermutation(features=feature_dim))
            
            mask = (torch.arange(0, feature_dim)>=(feature_dim//2)).float()
            # net_func = lambda in_d, out_d: MLP(in_shape=[in_d], out_shape=[out_d],
            #                                     hidden_sizes=[hidden_feature]*3, activation=F.leaky_relu)
            net_func = lambda in_d, out_d: ResidualNet(in_features=in_d, out_features=out_d,
                                                hidden_features=hidden_feature, context_features=context_feature,
                                                num_blocks=2, activation=F.leaky_relu, dropout_probability=0)
            transforms.append(AffineCouplingTransform(mask=mask, transform_net_create_fn=net_func))
        
        transform = CompositeTransform(transforms)
        base_dist = StandardNormal(shape=[feature_dim])
        flow = Flow(transform, base_dist)
        return flow

    def get_hmce_nflow_model(self,
                        feature_dim,
                        hidden_feature,
                        context_feature,
                        num_layers,
                        num_scales):
        flows = []
        permutations = []
        permutation_modes = ['identity', 'reverse', 'random']

        for scale_idx in range(num_scales):
            flows.append(self.get_1d_nflow_model(feature_dim, hidden_feature, context_feature, num_layers))
            mode = permutation_modes[scale_idx % len(permutation_modes)]
            permutations.append(FeaturePermutation(feature_dim, mode=mode, seed=scale_idx + 17))

        return MultiScaleFlow(flows, permutations, temperature=1.0)

    def _prototype_contrastive_loss(self, xa, labels, prototype_bank):
        if not prototype_bank:
            return torch.tensor(0.0, device=xa.device)

        available_labels = []
        prototypes = []
        for label, proto in prototype_bank.items():
            available_labels.append(int(label))
            prototypes.append(proto.to(xa.device))

        if len(prototypes) == 0:
            return torch.tensor(0.0, device=xa.device)

        label_to_idx = {label: idx for idx, label in enumerate(available_labels)}
        selected = [i for i, label in enumerate(labels.tolist()) if int(label) in label_to_idx]
        if len(selected) == 0:
            return torch.tensor(0.0, device=xa.device)

        proto_mat = torch.stack(prototypes, dim=0)
        xa_sel = F.normalize(xa[selected], dim=1)
        proto_mat = F.normalize(proto_mat, dim=1)
        logits = xa_sel @ proto_mat.t() / max(self.cpr_tau, 1e-6)
        target = torch.tensor([label_to_idx[int(labels[i].item())] for i in selected], device=xa.device)
        return F.cross_entropy(logits, target)

    def _maft_replay_scale(self, batch_acc, flow_prob_mean):
        if self.maft_gate is None:
            return torch.tensor(1.0, device=self.device)

        gate_input = torch.tensor(
            [[float(batch_acc), float(flow_prob_mean), float(self.flow_explore_theta)]],
            dtype=torch.float32,
            device=self.device,
        )
        return self.maft_gate(gate_input).squeeze()

    def get_extra_classifier_loss(self):
        """Hook for subclasses to add extra regularization to classifier loss.
        Override in CosineMixin to return EWC penalty."""
        return 0.0

    def train_a_batch(self,
                        x, y,
                        train_flow,
                        flow,
                        last_flow,
                        last_classifier,
                        global_classifier,
                        classes_so_far,
                        classes_past_task,
                        available_labels,
                        available_labels_past,
                        prototype_bank=None):
        
        # ===================
        # 1. prediction loss
        # ====================
        if not train_flow:
            return self.train_a_batch_classifier(x, y, flow, last_classifier, global_classifier, classes_past_task, available_labels, prototype_bank=prototype_bank)
        else:
            return self.train_a_batch_flow(x, y, last_flow, classes_so_far, available_labels_past)

    def sample_from_flow(self, flow, labels, batch_size):
        label = np.random.choice(labels, batch_size)
        class_onehot = np.zeros((batch_size, self.num_classes))
        class_onehot[np.arange(batch_size), label] = 1
        device = next(self.parameters()).device
        class_onehot = torch.Tensor(class_onehot).to(device)
        flow_xa = flow.sample(num_samples=1, context=class_onehot).squeeze(1)
        flow_xa = flow_xa.detach()
        return flow_xa, label, class_onehot

    def probability_in_localdata(self, xa_u, y, prob_mean, flow_xa, flow_label):
        flow_xa_label_set = set(flow_label)
        flow_xa_prob = torch.zeros([flow_xa.shape[0]], device=flow_xa.device)
        for flow_yi in flow_xa_label_set:
            if (y==flow_yi).sum()>0:
                xa_u_yi = xa_u[y==flow_yi]
                xa_u_yi_mean = torch.mean(xa_u_yi, dim=0, keepdim=True)
                xa_u_yi_var = torch.mean((xa_u_yi-xa_u_yi_mean)*(xa_u_yi-xa_u_yi_mean), dim=0, keepdim=True)

                flow_xa_yi = flow_xa[flow_label==flow_yi]
                prob_xa_yi_ = 1/np.sqrt(2*np.pi)*torch.pow(xa_u_yi_var+eps, -0.5)*torch.exp(-torch.pow(flow_xa_yi-xa_u_yi_mean, 2)*torch.pow(xa_u_yi_var+eps, -1)*0.5)
                prob_xa_yi = torch.mean(prob_xa_yi_, dim=1)
                flow_xa_prob[flow_label==flow_yi] = prob_xa_yi
            else:
                flow_xa_prob[flow_label==flow_yi] = prob_mean
        return flow_xa_prob
        
    def train_a_batch_classifier(self, x, y, flow, last_classifier, global_classifier, classes_past_task, available_labels, prototype_bank=None):
        device = self.device

        softmax_output, xa, logits = self.classifier(x)
        softmax_output = torch.clamp(softmax_output, min=eps, max=1.0)
        c_loss_cls = self.classify_criterion(torch.log(softmax_output), y)

        if self.algorithm == 'PreciseFCL':
            kd_loss_feature_last, kd_loss_output_last, kd_loss_feature_global, kd_loss_output_global = \
                self.knowledge_distillation_on_xa_output(x, xa, softmax_output, last_classifier, global_classifier)
            kd_loss_feature = (kd_loss_feature_last + kd_loss_feature_global) * self.k_kd_feature
            kd_loss_output = (kd_loss_output_last + kd_loss_output_global) * self.k_kd_output
            kd_loss = kd_loss_feature + kd_loss_output
        else:
            kd_loss_feature, kd_loss_output, kd_loss = 0, 0, 0

        current_proto_loss = self._prototype_contrastive_loss(xa, y, prototype_bank)
        c_loss = c_loss_cls + kd_loss + self.cpr_lambda * current_proto_loss

        correct = (torch.sum(torch.argmax(softmax_output, dim=1) == y)).item()
        batch_acc = float(correct) / max(float(x.shape[0]), 1.0)

        replay_loss_tensor = None
        flow_xa_prob_mean = 0.0
        prob_mean = 0.0
        kd_loss_flow = 0.0
        c_loss_flow = 0.0

        if self.algorithm == 'PreciseFCL' and type(flow) != type(None) and self.k_loss_flow > 0:
            batch_size = x.shape[0]

            with torch.no_grad():
                _, xa_detached, _ = self.classifier(x)
                xa_detached = xa_detached.reshape(xa_detached.shape[0], -1)

                y_one_hot = F.one_hot(y, num_classes=self.num_classes).float()
                log_prob, xa_u = flow.log_prob_and_noise(xa_detached, y_one_hot)
                log_prob = log_prob.detach()
                xa_u = xa_u.detach()
                prob_mean = torch.exp(log_prob / xa_detached.shape[1]).mean() + eps

                flow_xa, label, _ = self.sample_from_flow(flow, available_labels, batch_size)
                flow_xa_prob = self.probability_in_localdata(xa_u, y, prob_mean, flow_xa, label)
                flow_xa_prob = flow_xa_prob.detach()
                flow_xa_prob_mean = flow_xa_prob.mean()

            flow_xa = flow_xa.reshape(flow_xa.shape[0], *self.xa_shape)
            softmax_output_flow, _ = self.classifier.forward_from_xa(flow_xa)
            softmax_output_flow = torch.clamp(softmax_output_flow, min=eps, max=1.0)
            c_loss_flow_generate = (
                self.classify_criterion_noreduce(torch.log(softmax_output_flow), torch.Tensor(label).long().to(device)) * flow_xa_prob
            ).mean()
            k_loss_flow_explore_forget = (1 - self.flow_explore_theta) * prob_mean + self.flow_explore_theta

            if self.cpr:
                replay_proto_loss = self._prototype_contrastive_loss(
                    flow_xa.reshape(flow_xa.shape[0], -1),
                    torch.Tensor(label).long().to(device),
                    prototype_bank,
                )
            else:
                replay_proto_loss = torch.tensor(0.0, device=device)

            kd_loss_output_last_flow, kd_loss_output_global_flow = self.knowledge_distillation_on_output(
                flow_xa, softmax_output_flow, last_classifier, global_classifier
            )
            kd_loss_flow = (kd_loss_output_last_flow + kd_loss_output_global_flow) * self.k_kd_output

            replay_loss_tensor = (
                c_loss_flow_generate * k_loss_flow_explore_forget
                + kd_loss_flow
                + self.cpr_lambda * replay_proto_loss
            ) * self.k_loss_flow

            replay_scale = self._maft_replay_scale(batch_acc, float(flow_xa_prob_mean.detach().item()))
            replay_loss_tensor = replay_loss_tensor * replay_scale
            c_loss_flow = replay_loss_tensor

        if torch.isnan(c_loss) or torch.isinf(c_loss):
            logger.warning("NaN or Inf detected in current loss! Skipping this batch.")
            return {'c_loss': 0.0, 'kd_loss': 0.0, 'correct': 0, 'flow_prob_mean': 0.0,
                    'c_loss_flow': 0.0, 'kd_loss_flow': 0.0, 'kd_loss_feature': 0.0, 'kd_loss_output': 0.0}

        if replay_loss_tensor is not None and (torch.isnan(replay_loss_tensor) or torch.isinf(replay_loss_tensor)):
            logger.warning("NaN or Inf detected in replay loss! Dropping replay contribution for this batch.")
            replay_loss_tensor = None
            c_loss_flow = 0.0

        # Hook for extra regularization (e.g. EWC penalty from CosineMixin)
        extra_loss = self.get_extra_classifier_loss()
        if isinstance(extra_loss, torch.Tensor):
            c_loss = c_loss + extra_loss

        if self.gcar and replay_loss_tensor is not None:
            params = [param for param in self.classifier.parameters() if param.requires_grad]
            if self.maft_gate is not None:
                params += [param for param in self.maft_gate.parameters() if param.requires_grad]
            current_grads = torch.autograd.grad(c_loss, params, retain_graph=True, allow_unused=True)
            replay_grads = torch.autograd.grad(replay_loss_tensor, params, retain_graph=True, allow_unused=True)

            flat_current_list = []
            flat_replay_list = []
            for param, current_grad, replay_grad in zip(params, current_grads, replay_grads):
                if current_grad is None:
                    current_grad = torch.zeros_like(param)
                if replay_grad is None:
                    replay_grad = torch.zeros_like(param)
                flat_current_list.append(current_grad.reshape(-1))
                flat_replay_list.append(replay_grad.reshape(-1))

            flat_current = torch.cat(flat_current_list) if len(flat_current_list) > 0 else torch.tensor([], device=device)
            flat_replay = torch.cat(flat_replay_list) if len(flat_replay_list) > 0 else torch.tensor([], device=device)

            if flat_current.numel() > 0 and flat_replay.numel() > 0:
                dot = torch.dot(flat_current, flat_replay)
                norm_current = flat_current.norm().clamp_min(eps)
                norm_replay = flat_replay.norm().clamp_min(eps)
                beta = torch.sigmoid(dot / (norm_current * norm_replay + eps))
            else:
                dot = torch.tensor(0.0, device=device)
                beta = torch.tensor(1.0, device=device)

            final_grads = []
            for param, current_grad, replay_grad in zip(params, current_grads, replay_grads):
                if current_grad is None:
                    current_grad = torch.zeros_like(param)
                if replay_grad is None:
                    replay_grad = torch.zeros_like(param)
                if dot.item() < 0:
                    denom = current_grad.reshape(-1).dot(current_grad.reshape(-1)).clamp_min(eps)
                    replay_grad = replay_grad - (replay_grad.reshape(-1).dot(current_grad.reshape(-1)) / denom) * current_grad
                final_grads.append(current_grad + beta * replay_grad)

            self.classifier_optimizer.zero_grad()
            for param, grad in zip(params, final_grads):
                if grad is not None:
                    param.grad = grad
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0, norm_type='2')
            self.classifier_optimizer.step()
        else:
            if replay_loss_tensor is not None and (self.maft or self.cpr):
                self.classifier_optimizer.zero_grad()
                (c_loss + replay_loss_tensor).backward()
                clip_params = list(self.classifier.parameters())
                if self.maft_gate is not None:
                    clip_params += list(self.maft_gate.parameters())
                torch.nn.utils.clip_grad_norm_(clip_params, max_norm=1.0, norm_type='2')
                self.classifier_optimizer.step()
            else:
                self.classifier_optimizer.zero_grad()
                if hasattr(self, 'classifier_fb_optimizer'):
                    self.classifier_fb_optimizer.zero_grad()

                c_loss.backward(retain_graph=True)

                if replay_loss_tensor is not None:
                    fc2_params = [p for n, p in self.classifier.named_parameters() if 'fc2' in n and p.requires_grad]
                    if len(fc2_params) > 0:
                        replay_grads = torch.autograd.grad(replay_loss_tensor, fc2_params, retain_graph=True, allow_unused=True)
                        for p, g in zip(fc2_params, replay_grads):
                            if g is not None:
                                p.grad += g

                clip_params = list(self.classifier.parameters())
                if self.maft_gate is not None:
                    clip_params += list(self.maft_gate.parameters())
                torch.nn.utils.clip_grad_norm_(clip_params, max_norm=1.0, norm_type='2')
                self.classifier_optimizer.step()

        prob_mean = myitem(prob_mean)
        c_loss_flow = myitem(c_loss_flow)
        kd_loss = myitem(kd_loss)
        kd_loss_flow = myitem(kd_loss_flow)
        kd_loss_feature = myitem(kd_loss_feature)
        kd_loss_output = myitem(kd_loss_output)

        c_loss_val = c_loss.item()

        return {'c_loss': c_loss_val, 'kd_loss': kd_loss, 'correct': correct, 'flow_prob_mean': flow_xa_prob_mean,
                 'c_loss_flow': c_loss_flow, 'kd_loss_flow': kd_loss_flow, 'kd_loss_feature': kd_loss_feature, 'kd_loss_output': kd_loss_output}

    def knowledge_distillation_on_output(self, xa, softmax_output, last_classifier, global_classifier):
        if self.k_kd_last_cls>0 and type(last_classifier)!=type(None):
            softmax_output_last, _ = last_classifier.forward_from_xa(xa)
            softmax_output_last = softmax_output_last.detach()
            kd_loss_output_last = self.k_kd_last_cls*MultiClassCrossEntropy(softmax_output, softmax_output_last, T=2)
        else:
            kd_loss_output_last = 0

        if self.k_kd_global_cls>0:
            softmax_output_global, _ = global_classifier.forward_from_xa(xa)
            softmax_output_global = softmax_output_global.detach()
            kd_loss_output_global = self.k_kd_global_cls*MultiClassCrossEntropy(softmax_output, softmax_output_global, T=2)
        else:
            kd_loss_output_global = 0

        return kd_loss_output_last, kd_loss_output_global
    
    def knowledge_distillation_on_xa_output(self, x, xa, softmax_output, last_classifier, global_classifier):
        if self.k_kd_last_cls>0 and type(last_classifier)!=type(None):
            softmax_output_last, xa_last, _ = last_classifier(x)
            xa_last = xa_last.detach()
            softmax_output_last = softmax_output_last.detach()
            kd_loss_feature_last = self.k_kd_last_cls*torch.pow(xa_last-xa, 2).mean()
            kd_loss_output_last = self.k_kd_last_cls*MultiClassCrossEntropy(softmax_output, softmax_output_last, T=2)
        else:
            kd_loss_feature_last = 0
            kd_loss_output_last = 0
        
        if self.k_kd_global_cls>0:
            softmax_output_global, xa_global, _ = global_classifier(x)
            xa_global = xa_global.detach()
            softmax_output_global = softmax_output_global.detach()
            kd_loss_feature_global = self.k_kd_global_cls*torch.pow(xa_global-xa, 2).mean()
            kd_loss_output_global = self.k_kd_global_cls*MultiClassCrossEntropy(softmax_output, softmax_output_global, T=2)
        else:
            kd_loss_feature_global = 0
            kd_loss_output_global = 0
        
        return kd_loss_feature_last, kd_loss_output_last, kd_loss_feature_global, kd_loss_output_global

    def train_a_batch_flow(self, x, y, last_flow, classes_so_far, available_labels_past):            
        xa = self.classifier.forward_to_xa(x)
        xa = xa.reshape(xa.shape[0], -1)
        y_one_hot = F.one_hot(y, num_classes=self.num_classes).float()
        loss_data = -self.flow.log_prob(inputs=xa, context=y_one_hot).mean()
        
        # Check for NaN in loss_data
        if torch.isnan(loss_data) or torch.isinf(loss_data):
            loss_data = torch.tensor(0.0, device=self.device, requires_grad=True)

        if self.algorithm == 'PreciseFCL' and type(last_flow)!=type(None):
            batch_size = x.shape[0]
            with torch.no_grad():
                flow_xa, label, label_one_hot = self.sample_from_flow(last_flow, available_labels_past, batch_size)
            loss_last_flow = -self.flow.log_prob(inputs=flow_xa, context=label_one_hot).mean()
        else:
            loss_last_flow = 0
        loss_last_flow = self.k_flow_lastflow*loss_last_flow

        loss = loss_data + loss_last_flow

        self.flow_optimizer.zero_grad()
        loss.backward()
        self.flow_optimizer.step()

        return {'flow_loss': loss_data.item(), 'flow_loss_last': myitem(loss_last_flow)}
