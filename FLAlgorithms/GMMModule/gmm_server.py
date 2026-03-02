"""
GMMServerPreciseFCL — FedPrecise with GMMUserPreciseFCL clients.

The ONLY override is init_users(), which instantiates GMMUserPreciseFCL
instead of UserPreciseFCL.  All aggregation, CL task scheduling, model
broadcast, and evaluation logic is inherited unchanged from FedPrecise.
"""

import glog as logger

from FLAlgorithms.servers.serverPreciseFCL import FedPrecise
from FLAlgorithms.GMMModule.gmm_model import GMMPreciseModel
from FLAlgorithms.GMMModule.gmm_user import GMMUserPreciseFCL
from utils.model_utils import read_user_data_PreciseFCL


class GMMServerPreciseFCL(FedPrecise):

    def __init__(self, args, model: GMMPreciseModel, seed):
        # FedPrecise.__init__ sets classifier_head_list BEFORE calling
        # self.init_users(), so our override sees the correct value.
        super().__init__(args, model, seed)

    # ------------------------------------------------------------------
    # User initialisation — identical to FedPrecise except user class
    # ------------------------------------------------------------------

    def init_users(self, data, args, model: GMMPreciseModel):
        self.users = []
        total_users = len(data['client_names'])

        for i in range(total_users):
            id, train_data, test_data, label_info = read_user_data_PreciseFCL(
                i, data,
                dataset=args.dataset,
                count_labels=True,
                task=0,
            )

            self.total_train_samples += len(train_data)
            self.total_test_samples  += len(test_data)

            user = GMMUserPreciseFCL(
                args,
                i,
                model,
                train_data,
                test_data,
                label_info,
                use_adam=self.use_adam,
                my_model_name='gmm-fedprecise',
                unique_labels=self.unique_labels,
                classifier_head_list=self.classifier_head_list,
            )
            self.users.append(user)

            # Seed CL label bookkeeping for task 0 (same as FedPrecise)
            user.classes_so_far.extend(label_info['labels'])
            user.current_labels.extend(label_info['labels'])

        logger.info(
            "Number of Train/Test samples: %d/%d",
            self.total_train_samples, self.total_test_samples,
        )
        logger.info("Data from %d users in total.", total_users)
        logger.info("Finished creating GMMServerPreciseFCL.")
