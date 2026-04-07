import copy

import glog as logger
import numpy as np
import torch

from FLAlgorithms.GMMModule.gmm_user import GMMUserPreciseFCL
from FLAlgorithms.servers.serverPreciseFCL import FedPrecise
from FLAlgorithms.users.userPreciseFCL import UserPreciseFCL
from utils.model_utils import read_user_data_PreciseFCL


class ACTAServerPreciseFCL(FedPrecise):
    def __init__(self, args, model, seed):
        self.prototype_bank = {}
        super().__init__(args, model, seed)

    def init_users(self, data, args, model):
        self.users = []
        total_users = len(data['client_names'])

        for i in range(total_users):
            user_id, train_data, test_data, label_info = read_user_data_PreciseFCL(
                i, data, dataset=args.dataset, count_labels=True, task=0
            )

            self.total_train_samples += len(train_data)
            self.total_test_samples += len(test_data)

            if getattr(args, 'gmm', False):
                user = GMMUserPreciseFCL(
                    args,
                    i,
                    model,
                    train_data,
                    test_data,
                    label_info,
                    use_adam=self.use_adam,
                    my_model_name='acta-gmm-fedprecise',
                    unique_labels=self.unique_labels,
                    classifier_head_list=self.classifier_head_list,
                )
            else:
                user = UserPreciseFCL(
                    args,
                    i,
                    model,
                    train_data,
                    test_data,
                    label_info,
                    use_adam=self.use_adam,
                    my_model_name='acta-fedprecise',
                    unique_labels=self.unique_labels,
                    classifier_head_list=self.classifier_head_list,
                )

            self.users.append(user)
            user.classes_so_far.extend(label_info['labels'])
            user.current_labels.extend(label_info['labels'])

        logger.info(
            'Number of Train/Test samples: %d/%d',
            self.total_train_samples,
            self.total_test_samples,
        )
        logger.info('Data from %d users in total.', total_users)
        logger.info('Finished creating ACTAServerPreciseFCL.')

    def _get_user_embedding(self, user):
        embedding = user.get_task_embedding()
        if embedding is None:
            return None
        if torch.is_tensor(embedding):
            return embedding.detach().to(self.device)
        return torch.tensor(embedding, dtype=torch.float32, device=self.device)

    def _update_prototype_bank(self, users):
        momentum = getattr(self.args, 'acta_momentum', 0.9)
        for user in users:
            embedding = self._get_user_embedding(user)
            if embedding is None:
                continue
            for label in user.current_labels:
                label = int(label)
                old_proto = self.prototype_bank.get(label)
                if old_proto is None:
                    self.prototype_bank[label] = embedding.detach().clone()
                else:
                    self.prototype_bank[label] = momentum * old_proto + (1.0 - momentum) * embedding.detach().clone()

    def aggregate_parameters_(self, class_partial):
        assert self.selected_users is not None and len(self.selected_users) > 0

        embeddings = []
        valid_users = []
        for user in self.selected_users:
            embedding = self._get_user_embedding(user)
            if embedding is None:
                continue
            embeddings.append(embedding)
            valid_users.append(user)

        if len(valid_users) == 0:
            logger.warning('ACTA could not compute client embeddings; falling back to FedAvg.')
            return super().aggregate_parameters_(class_partial)

        if len(self.prototype_bank) > 0:
            p_global = torch.stack(list(self.prototype_bank.values())).mean(dim=0)
        else:
            p_global = torch.stack(embeddings).mean(dim=0)

        tau = max(float(getattr(self.args, 'acta_tau', 0.5)), 1e-6)
        similarities = torch.stack([
            torch.nn.functional.cosine_similarity(emb.unsqueeze(0), p_global.unsqueeze(0), dim=1).squeeze(0)
            for emb in embeddings
        ])
        attention = torch.softmax(similarities / tau, dim=0)

        logger.info('ACTA attention weights: %s', attention.detach().cpu().numpy().round(4).tolist())

        param_dict = {}
        for name, param in self.model.named_parameters():
            param_dict[name] = torch.zeros_like(param.data)

        for weight, user in zip(attention, valid_users):
            for name, param in user.model.named_parameters():
                param_dict[name] += param.data * weight

        for name, param in self.model.named_parameters():
            param.data = param_dict[name].clone()

        self._update_prototype_bank(valid_users)
