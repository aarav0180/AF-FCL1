"""
GMMUserPreciseFCL — UserPreciseFCL with GMM prior fitting hooked into next_task().

The ONLY override is next_task().  Everything else (train, set_parameters,
test_all_) is inherited verbatim from UserPreciseFCL.

Execution order inside next_task()
------------------------------------
1. fit_gmm_prior(trainloaderfull, device)
       ↳ runs task-T features through the flow transform
       ↳ fits K Gaussian components in latent z-space
       ↳ stores params as GPU tensors inside TaskGMMPrior

2. super().next_task(...)
       ↳ self.last_copy = deepcopy(self.model)     ← captures fitted GMM
       ↳ self.trainloaderfull updated to task T+1
       ↳ CL label bookkeeping advanced

Result
------
When task T+1's flow trains:
    loss = -flow.log_prob(xa_new | y_new).mean()
         = -log p_GMM(z_new) - log|det J_f|

p_GMM is the task-T cluster structure, so the new task's latent codes are
regularised to stay near the multi-modal geometry of the previous task.
"""

import torch

from FLAlgorithms.users.userPreciseFCL import UserPreciseFCL
from FLAlgorithms.GMMModule.gmm_model import GMMPreciseModel


class GMMUserPreciseFCL(UserPreciseFCL):

    def __init__(
        self,
        args,
        id,
        model: GMMPreciseModel,
        train_data,
        test_data,
        label_info,
        use_adam=False,
        my_model_name=None,
        unique_labels=None,
        classifier_head_list=None,
    ):
        if classifier_head_list is None:
            classifier_head_list = []
        super().__init__(
            args, id, model, train_data, test_data,
            label_info,
            use_adam=use_adam,
            my_model_name=my_model_name,
            unique_labels=unique_labels,
            classifier_head_list=classifier_head_list,
        )

    # ------------------------------------------------------------------
    # CL task transition — fit GMM BEFORE freezing the model
    # ------------------------------------------------------------------

    def next_task(self, train, test, label_info=None, if_label=True):
        """
        1. Fit the GMM prior on the CURRENT task's latent codes (task T).
           self.trainloaderfull still points to task-T data at this point.
        2. Call super().next_task() which:
             - deepcopies the model into self.last_copy  ← captures fitted GMM
             - updates self.trainloaderfull to task T+1 data
             - advances CL label bookkeeping
        """
        # Resolve device (mirrors userPreciseFCL pattern)
        if self.args.device == 'cuda' and torch.cuda.is_available():
            device = torch.device('cuda:0')
        else:
            device = torch.device('cpu')

        # Fit GMM on task-T features before the model is frozen
        if isinstance(self.model, GMMPreciseModel):
            self.model.fit_gmm_prior(self.trainloaderfull, device)

        # Advance task (deepcopy + data update + label bookkeeping)
        super().next_task(train, test, label_info=label_info, if_label=if_label)
