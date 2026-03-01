"""
SCAFFOLD server implementation for AF-FCL.

Reference: Karimireddy et al., "SCAFFOLD: Stochastic Controlled Averaging for
Federated Learning", ICML 2020.

The server maintains a global control variate c (list of tensors, one per
model parameter).  After each round it aggregates the client deltas:

    c  +=  (|S| / N) * mean_over_S(delta_c_i)

Since this codebase selects ALL clients every round, |S| = N and the update
simplifies to the mean of delta_c_i across all clients.

The continual-learning task loop is identical to FedPrecise.train() — only the
local update rule (inside UserSCAFFOLD) and the aggregation step change.
"""

import copy
import time

import glog as logger
import numpy as np
import torch

from FLAlgorithms.servers.serverbase import Server
from FLAlgorithms.users.userSCAFFOLD import UserSCAFFOLD
from FLAlgorithms.PreciseFCLNet.model import PreciseModel
from utils.dataset import get_dataset
from utils.model_utils import read_user_data_PreciseFCL


class ServerSCAFFOLD(Server):
    def __init__(self, args, model: PreciseModel, seed):
        super().__init__(args, model, seed)

        self.use_adam = 'adam' in self.algorithm.lower()
        self.data = get_dataset(args, args.dataset, args.datadir, args.data_split_file)
        self.unique_labels = self.data['unique_labels']
        self.init_users(self.data, args, model)

        # Resolve device
        if args.device == 'cuda':
            if torch.cuda.is_available():
                device = torch.device('cuda:0')
            else:
                logger.warning("CUDA requested but not available, falling back to CPU")
                device = torch.device('cpu')
        else:
            device = torch.device('cpu')
        logger.info('Using device: ' + str(device))
        self.device = device

        # Move models to device and initialise per-user control variates on the
        # correct device (done here so tensors match model parameter devices).
        # Control variates cover classifier parameters only — the flow is
        # trained with its own optimizer and does not need SCAFFOLD correction.
        for u in self.users:
            u.model = u.model.to(device)
            u.c_i      = [torch.zeros_like(p) for p in u.model.classifier.parameters()]
            u.server_c = [torch.zeros_like(p) for p in u.model.classifier.parameters()]

        # Move server model and initialise global control variate
        self.model.to(device)
        self.c = [torch.zeros_like(p) for p in self.model.classifier.parameters()]

    # ------------------------------------------------------------------
    # User initialisation
    # ------------------------------------------------------------------

    def init_users(self, data, args, model: PreciseModel):
        self.users = []
        total_users = len(data['client_names'])
        for i in range(total_users):
            id, train_data, test_data, label_info = read_user_data_PreciseFCL(
                i, data, dataset=args.dataset, count_labels=True, task=0)

            self.total_train_samples += len(train_data)
            self.total_test_samples += len(test_data)

            user = UserSCAFFOLD(
                args,
                i,
                model,
                train_data,
                test_data,
                label_info,
                unique_labels=self.unique_labels,
            )
            self.users.append(user)

            # Initialise CL label bookkeeping (same as FedPrecise)
            user.classes_so_far.extend(label_info['labels'])
            user.current_labels.extend(label_info['labels'])

        logger.info("Number of Train/Test samples: %d/%d" % (
            self.total_train_samples, self.total_test_samples))
        logger.info("Data from {} users in total.".format(total_users))
        logger.info("Finished creating SCAFFOLD server.")

    # ------------------------------------------------------------------
    # Control-variate broadcast
    # ------------------------------------------------------------------

    def send_server_cv(self):
        """Push a copy of the global control variate c to every user."""
        for user in self.users:
            user.server_c = [c.clone() for c in self.c]

    # ------------------------------------------------------------------
    # Training loop  (same CL task structure as FedPrecise.train)
    # ------------------------------------------------------------------

    def train(self, args):
        N_TASKS = len(
            self.data['train_data'][self.data['client_names'][0]]['x'])

        for task in range(N_TASKS):

            # ==========================================
            # Task 0 — initialise label-tracking fields
            # ==========================================
            if task == 0:
                available_labels = set()
                available_labels_current = set()
                available_labels_past = set()
                for u in self.users:
                    available_labels = available_labels.union(
                        set(u.classes_so_far))
                    available_labels_current = available_labels_current.union(
                        set(u.current_labels))
                for u in self.users:
                    u.available_labels = list(available_labels)
                    u.available_labels_current = list(available_labels_current)
                    u.available_labels_past = list(available_labels_past)

            # ==========================================
            # Task > 0 — advance each user to new task
            # ==========================================
            else:
                self.current_task = task

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                for i in range(len(self.users)):
                    id, train_data, test_data, label_info = \
                        read_user_data_PreciseFCL(
                            i, self.data,
                            dataset=args.dataset,
                            count_labels=True,
                            task=task)
                    self.users[i].next_task(train_data, test_data, label_info)

                # Refresh global label sets
                available_labels = set()
                available_labels_current = set()
                available_labels_past = self.users[0].available_labels
                for u in self.users:
                    available_labels = available_labels.union(
                        set(u.classes_so_far))
                    available_labels_current = available_labels_current.union(
                        set(u.current_labels))
                for u in self.users:
                    u.available_labels = list(available_labels)
                    u.available_labels_current = list(available_labels_current)
                    u.available_labels_past = list(available_labels_past)

            # Log label info
            for u in self.users:
                logger.info("classes so far: " + str(u.classes_so_far))
            logger.info("available labels for the Client: " +
                        str(self.users[-1].available_labels))
            logger.info("available labels (current) for the Client: " +
                        str(self.users[-1].available_labels_current))

            # ==========================================
            # Round loop
            # ==========================================
            epoch_per_task = int(self.num_glob_iters / N_TASKS)

            for glob_iter_task in range(epoch_per_task):
                glob_iter = glob_iter_task + epoch_per_task * task

                logger.info(
                    "\n\n------------- Round number: %d | "
                    "Current task: %d -------------\n\n"
                    % (glob_iter, task))

                # Select users (all users every round in this codebase)
                self.selected_users, self.user_idxs = self.select_users(
                    glob_iter, len(self.users), return_idx=True)

                # Broadcast server model → clients (also snapshots x_global)
                self.send_parameters(mode='all', beta=1)

                # Broadcast global control variate → clients
                self.send_server_cv()

                self.timestamp = time.time()
                self.pickle_record['train'][glob_iter] = {}

                global_classifier = self.model.classifier
                global_classifier.eval()

                # ----- local training -----
                for user_id, user in zip(
                        self.user_idxs, self.selected_users):
                    user_result = user.train(
                        glob_iter,
                        glob_iter_task,
                        global_classifier,
                        verbose=True,
                    )
                    self.pickle_record['train'][glob_iter][user_id] = \
                        user_result

                curr_timestamp = time.time()
                train_time = (
                    (curr_timestamp - self.timestamp) /
                    len(self.selected_users))
                self.metrics['user_train_time'].append(train_time)

                self.timestamp = time.time()

                # ----- SCAFFOLD aggregation -----
                self.aggregate_parameters_scaffold()

                curr_timestamp = time.time()
                agg_time = curr_timestamp - self.timestamp
                self.metrics['server_agg_time'].append(agg_time)

            # Broadcast final model after task completes
            self.send_parameters(mode='all', beta=1)

            self.evaluate_all_(
                glob_iter=glob_iter, matrix=True, personal=False)
            self.save_pickle()

    # ------------------------------------------------------------------
    # SCAFFOLD aggregation
    # ------------------------------------------------------------------

    def aggregate_parameters_scaffold(self):
        """
        Two-step aggregation per SCAFFOLD:

        1. Weighted-average model parameters (identical to FedAvg).
        2. Update global control variate:
               c += (|S| / N) * mean_over_S(delta_c_i)
           Since |S| = N (all clients selected every round):
               c += mean(delta_c_i)
        """
        assert self.selected_users is not None and len(self.selected_users) > 0

        # ---- 1. FedAvg model weights --------------------------------
        param_dict = {}
        for name, param in self.model.named_parameters():
            param_dict[name] = torch.zeros_like(param.data)

        total_train = sum(u.train_samples for u in self.selected_users)

        for user in self.selected_users:
            for name, param in user.model.named_parameters():
                param_dict[name] += (
                    param.data * user.train_samples / total_train)

        for name, param in self.model.named_parameters():
            param.data = param_dict[name]

        # ---- 2. Update global control variate -----------------------
        # SCAFFOLD formula:  c += (|S|/N) * (1/|S|) * sum_S(delta_c_i)
        #                      = (1/N) * sum_S(delta_c_i)
        # When |S| = N:       = mean(delta_c_i)
        N = len(self.users)
        with torch.no_grad():
            for j, c_g in enumerate(self.c):
                c_delta_sum = torch.zeros_like(c_g)
                for user in self.selected_users:
                    c_delta_sum += user.delta_c[j]
                c_g.data += c_delta_sum / N
