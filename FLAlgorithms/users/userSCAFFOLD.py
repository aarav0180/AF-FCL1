"""
SCAFFOLD user with AF-FCL anti-forgetting (normalizing flow + generative replay).

SCAFFOLD (Karimireddy et al., ICML 2020) corrects client drift:
  - Each local gradient step injects (server_c - c_i) via a backward hook,
    correcting the classifier gradient toward the global optimum.
  - After K local steps, c_i is updated via the weight-difference formula
    (Option II) and delta_c is stored for server aggregation.

The AF-FCL continual-learning mechanisms are kept intact:
  - The normalizing flow is trained each iteration (models per-task feature
    distributions for generative replay).
  - The classifier is trained on both real data and flow-generated past-task
    features (k_loss_flow > 0 path inside model.train_a_batch).

The SCAFFOLD correction is injected via backward hooks on classifier parameters
so that model.train_a_batch (which calls optimizer.step() internally) picks up
the corrected gradients without any changes to the model's training logic.
The control variate (c_i / server_c) covers classifier parameters only;
the flow is trained with its own separate optimizer without correction.
"""

import copy

import glog as logger
import torch
from torch.utils.data import DataLoader

from FLAlgorithms.PreciseFCLNet.model import PreciseModel
from FLAlgorithms.users.userbase import User
from utils.meter import Meter

eps = 1e-30


class UserSCAFFOLD(User):
    def __init__(
        self,
        args,
        id,
        model: PreciseModel,
        train_data,
        test_data,
        label_info,
        unique_labels=None,
    ):
        super().__init__(
            args, id, model, train_data, test_data,
            use_adam=True,
            my_model_name="scaffold",
            unique_labels=unique_labels,
        )

        self.label_info = label_info
        self.k_loss_flow = args.k_loss_flow
        self.use_lastflow_x = args.use_lastflow_x

        # SCAFFOLD control variates — initialised as empty lists.
        # ServerSCAFFOLD.__init__ fills them after moving the model to the
        # correct device (ensures tensors are on the right device from the start).
        # c_i / server_c cover classifier parameters only (not flow).
        self.c_i: list = []          # local control variate
        self.server_c: list = []     # copy of server's global control variate
        self.delta_c = None          # c_i_new - c_i_old, read by server after round
        self.x_global = None         # snapshot of global classifier params before training

    # ------------------------------------------------------------------
    # CL helpers
    # ------------------------------------------------------------------

    def next_task(self, train, test, label_info=None, if_label=True):
        """Advance to the next continual-learning task."""
        # Snapshot current model as last_copy for flow replay and KD
        self.last_copy = copy.deepcopy(self.model)
        if self.args.device == 'cuda' and torch.cuda.is_available():
            device = torch.device('cuda:0')
        else:
            device = torch.device('cpu')
        self.last_copy.to(device)
        self.if_last_copy = True

        # Update dataset
        self.train_data = train
        self.test_data = test
        self.train_samples = len(self.train_data)
        self.test_samples = len(self.test_data)

        self.trainloader = DataLoader(
            self.train_data, self.batch_size, drop_last=True, shuffle=True)
        self.testloader = DataLoader(
            self.test_data, self.batch_size, drop_last=True)
        self.testloaderfull = DataLoader(self.test_data, len(self.test_data))
        self.trainloaderfull = DataLoader(
            self.train_data, len(self.train_data), shuffle=True)
        self.iter_trainloader = iter(self.trainloader)
        self.iter_testloader = iter(self.testloader)

        # Update CL label bookkeeping
        self.classes_past_task = copy.deepcopy(self.classes_so_far)
        if if_label:
            self.classes_so_far.extend(label_info['labels'])
            self.current_labels.clear()
            self.current_labels.extend(label_info['labels'])

        self.test_data_so_far_loader.append(DataLoader(self.test_data, 64))
        self.test_data_per_task.append(self.test_data)
        self.current_task += 1

    # ------------------------------------------------------------------
    # Parameter sync  (called by server.send_parameters)
    # ------------------------------------------------------------------

    def set_parameters(self, model: PreciseModel, beta=1):
        """
        Copy global model weights into local model, then snapshot classifier
        params as x_global so the Option II c_i update can compute y_i - x_global
        after local training.
        """
        for old_param, new_param, local_param in zip(
                self.model.parameters(), model.parameters(), self.local_model):
            if beta == 1:
                old_param.data = new_param.data.clone()
                local_param.data = new_param.data.clone()
            else:
                old_param.data = (
                    beta * new_param.data.clone()
                    + (1 - beta) * old_param.data.clone()
                )
                local_param.data = (
                    beta * new_param.data.clone()
                    + (1 - beta) * local_param.data.clone()
                )

        # Snapshot global classifier params (x) for Option II.
        # c_i covers classifier params only — snapshot only those.
        self.x_global = [
            p.data.clone() for p in self.model.classifier.parameters()]

    # ------------------------------------------------------------------
    # Local training
    # ------------------------------------------------------------------

    def train(self, glob_iter, glob_iter_task, global_classifier, verbose):
        """
        Run K = local_epochs iterations combining AF-FCL flow-based replay
        with SCAFFOLD gradient correction.

        Each iteration:
          1. [Flow step]  Train the normalizing flow on current-task features
             (same as UserPreciseFCL when k_loss_flow > 0).  No SCAFFOLD
             correction here — the flow uses its own optimizer.
          2. [Classifier step]  Train the classifier via model.train_a_batch,
             which handles cross-entropy + flow-based replay of past features.
             SCAFFOLD correction (server_c[j] - c_i[j]) is injected into each
             classifier parameter's gradient via a backward hook, so it is
             applied between loss.backward() and optimizer.step() — without
             touching any logic inside model.train_a_batch.

        After all K steps, update c_i via Option II (no extra backward pass).
        """
        if self.args.device == 'cuda' and torch.cuda.is_available():
            device = torch.device('cuda:0')
        else:
            device = torch.device('cpu')

        correct = 0
        sample_num = 0
        cls_meter = Meter()

        for _ in range(self.local_epochs):
            samples = self.get_next_train_batch(count_labels=True)
            x, y = samples['X'].to(device), samples['y'].to(device)

            last_classifier = None
            last_flow = None
            if type(self.last_copy) != type(None):
                last_classifier = self.last_copy.classifier
                last_classifier.eval()
                last_flow = self.last_copy.flow
                if last_flow is not None:
                    last_flow.eval()

            # ---- Flow training step (AF-FCL anti-forgetting) ----
            if self.k_loss_flow > 0:
                self.model.classifier.eval()
                self.model.flow.train()
                flow_result = self.model.train_a_batch(
                    x, y,
                    train_flow=True,
                    flow=None,
                    last_flow=last_flow,
                    last_classifier=last_classifier,
                    global_classifier=global_classifier,
                    classes_so_far=self.classes_so_far,
                    classes_past_task=self.classes_past_task,
                    available_labels=self.available_labels,
                    available_labels_past=self.available_labels_past)
                cls_meter._update(flow_result, batch_size=x.shape[0])

            # ---- Choose flow for classifier replay ----
            flow = None
            if self.use_lastflow_x:
                flow = last_flow
            else:
                if self.model.flow is not None:
                    flow = self.model.flow
                    flow.eval()

            # ---- Classifier step with SCAFFOLD correction ----
            self.model.classifier.train()

            # Register backward hooks on every classifier parameter.
            # During loss.backward() inside model.train_a_batch, each hook
            # adds (server_c[j] - c_i[j]) to the raw gradient so the
            # subsequent optimizer.step() sees the SCAFFOLD-corrected gradient.
            hooks = []
            for p, ci_j, sc_j in zip(
                    self.model.classifier.parameters(),
                    self.c_i, self.server_c):
                correction = (sc_j - ci_j).detach().clone()
                h = p.register_hook(
                    lambda grad, corr=correction: grad + corr)
                hooks.append(h)

            cls_result = self.model.train_a_batch(
                x, y,
                train_flow=False,
                flow=flow,
                last_flow=last_flow,
                last_classifier=last_classifier,
                global_classifier=global_classifier,
                classes_so_far=self.classes_so_far,
                classes_past_task=self.classes_past_task,
                available_labels=self.available_labels,
                available_labels_past=self.available_labels_past)

            for h in hooks:
                h.remove()

            correct += cls_result['correct']
            sample_num += x.shape[0]
            cls_meter._update(cls_result, batch_size=x.shape[0])

        acc = float(correct) / sample_num
        result_dict = cls_meter.get_scalar_dict('global_avg')
        if 'flow_loss' not in result_dict:
            result_dict['flow_loss'] = 0
        if 'flow_loss_last' not in result_dict:
            result_dict['flow_loss_last'] = 0

        if verbose:
            logger.info(
                "SCAFFOLD Training for user {:d};  "
                "Acc: {:.2f}%%;  c_loss: {:.4f};  "
                "flow_loss: {:.4f};  flow_loss_last: {:.4f}".format(
                    self.id, acc * 100.0,
                    result_dict['c_loss'],
                    result_dict['flow_loss'],
                    result_dict['flow_loss_last']))

        # ------------------------------------------------------------------
        # Option II control-variate update (no extra backward pass needed)
        #
        #   coef    = 1 / (K * eta_l)
        #   y_delta = y_i - x_global       (local after training - global before)
        #   c_i_new = c_i - server_c - coef * y_delta
        #   delta_c = c_i_new - c_i
        #
        # c_i covers classifier parameters only.
        # ------------------------------------------------------------------
        with torch.no_grad():
            coef = 1.0 / (self.local_epochs * self.args.lr)
            c_plus = []
            self.delta_c = []
            for ci_j, sc_j, x_g_j, y_i_j in zip(
                    self.c_i, self.server_c,
                    self.x_global,
                    self.model.classifier.parameters()):
                y_delta = y_i_j.data - x_g_j
                c_plus_j = ci_j - sc_j - coef * y_delta
                self.delta_c.append(c_plus_j - ci_j)
                c_plus.append(c_plus_j)
            self.c_i = c_plus

        return {
            'acc': acc,
            'c_loss': result_dict['c_loss'],
            'flow_loss': result_dict['flow_loss'],
            'flow_loss_last': result_dict['flow_loss_last'],
        }

    # ------------------------------------------------------------------
    # Evaluation  (identical to UserPreciseFCL.test_all_)
    # ------------------------------------------------------------------

    def test_all_(self, personal=False, matrix=False):
        """Evaluate on all tasks seen so far (called by server.evaluate_all_)."""
        model = self.model.classifier

        if self.args.device == 'cuda' and torch.cuda.is_available():
            device = torch.device('cuda:0')
        else:
            device = torch.device('cpu')
        model.to(device)
        model.eval()

        predicts = []
        labels = []
        task_losses = []
        task_accs = []
        task_samples = []

        with torch.no_grad():
            for test_loader in self.test_data_so_far_loader:
                loss = 0.0
                test_correct = 0
                num_samples = 0
                for x, y in test_loader:
                    x = x.to(device)
                    y = y.to(device)
                    p, _, _ = model(x)
                    loss += self.model.classify_criterion(
                        torch.log(p + eps), y).item()
                    test_correct += (
                        torch.argmax(p, dim=1) == y).sum().item()
                    num_samples += y.shape[0]
                    if matrix:
                        predicts += torch.argmax(p, dim=1).cpu().tolist()
                        labels += y.cpu().tolist()

                task_losses.append(loss / num_samples)
                task_accs.append(float(test_correct) / num_samples)
                task_samples.append(num_samples)

        if matrix:
            return task_accs, task_losses, task_samples, predicts, labels
        return task_accs, task_losses, task_samples

