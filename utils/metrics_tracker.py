"""
ResearchMetricsTracker — central orchestrator for all research metrics
in the AF-FCL federated continual learning pipeline.

Tracks:
  1. Task accuracy matrix A[t][i] (t = task learned, i = task evaluated)
  2. Forgetting: F_i = max(A[0..t-1][i]) - A[t][i]
  3. Backward Transfer (BWT)
  4. Forward Transfer (FWT)
  5. Communication cost per round and cumulative
  6. Cosine similarity statistics (delegated to CosineTracker)

Exports:
  - forgetting_metrics.csv
  - communication_metrics.csv
  - research_metrics_summary.json

All tracking is observational — no training logic is modified.
"""

import os
import json
import time
import numpy as np
import pandas as pd
import torch
import glog as logger


class ResearchMetricsTracker:
    """
    Central metrics tracker initialised once in the server and updated
    at well-defined hook points during training.

    Parameters
    ----------
    output_dir : str
        Directory for CSV/JSON/plot output.
    num_tasks : int
        Total number of continual-learning tasks.
    num_clients : int
        Number of federated clients.
    tb_log : bool
        If True, also log scalars to TensorBoard.
    """

    def __init__(self, output_dir: str, num_tasks: int, num_clients: int,
                 tb_log: bool = False):
        self.output_dir = output_dir
        self.num_tasks = num_tasks
        self.num_clients = num_clients
        self.plots_dir = os.path.join(output_dir, 'plots')
        os.makedirs(self.plots_dir, exist_ok=True)

        # ------------------------------------------------------------------
        # 1. Accuracy matrix  A[t][i]
        #    Row t = state after learning task t
        #    Col i = accuracy on task i
        #    Unfilled entries are NaN (task i not yet seen at time t)
        # ------------------------------------------------------------------
        self.accuracy_matrix = np.full((num_tasks, num_tasks), np.nan)

        # Per-user accuracy matrix (averaged across users per call)
        self.user_task_acc_history = []  # list of np.ndarray per task

        # Global accuracy after each task
        self.glob_acc_per_task = []

        # ------------------------------------------------------------------
        # 2. Communication cost
        # ------------------------------------------------------------------
        self.comm_rounds = []  # list of dicts per round

        # ------------------------------------------------------------------
        # 3. Cosine similarity (accumulated by CosineTracker, stored here)
        # ------------------------------------------------------------------
        self.cosine_stats_per_task = {}  # task_idx -> dict of stats

        # ------------------------------------------------------------------
        # 4. Fairness results (populated after final evaluation)
        # ------------------------------------------------------------------
        self.fairness_results = {}

        # ------------------------------------------------------------------
        # 5. TensorBoard writer (optional)
        # ------------------------------------------------------------------
        self.tb_writer = None
        if tb_log:
            try:
                from torch.utils.tensorboard import SummaryWriter
                tb_dir = os.path.join(output_dir, 'tb_logs')
                self.tb_writer = SummaryWriter(log_dir=tb_dir)
                logger.info("TensorBoard logging enabled → %s", tb_dir)
            except ImportError:
                logger.warning("tensorboard not installed; skipping TB logging")

        # Timing
        self._start_time = time.time()

    # ======================================================================
    #  1. ACCURACY & FORGETTING
    # ======================================================================

    def record_task_accuracy(self, task_idx: int, task_accs: np.ndarray,
                             user_task_acc: np.ndarray = None,
                             glob_acc: float = None):
        """
        Record per-task accuracy after completing task `task_idx`.

        Parameters
        ----------
        task_idx : int
            Index of the task just completed (0-based).
        task_accs : np.ndarray, shape [num_tasks_seen_so_far]
            Weighted-average accuracy on each task seen so far.
        user_task_acc : np.ndarray, shape [num_users, num_tasks_seen_so_far]
            Per-user per-task accuracies (optional).
        glob_acc : float
            Overall global accuracy (optional).
        """
        num_tasks_seen = len(task_accs)
        self.accuracy_matrix[task_idx, :num_tasks_seen] = task_accs

        if user_task_acc is not None:
            self.user_task_acc_history.append(user_task_acc.copy())

        if glob_acc is not None:
            self.glob_acc_per_task.append(glob_acc)

        # Compute and log forgetting for this task
        forgetting = self._compute_forgetting(task_idx)
        avg_forgetting = np.nanmean(forgetting) if len(forgetting) > 0 else 0.0

        logger.info("[ResearchMetrics] Task %d complete | "
                    "Avg Forgetting: %.4f | Global Acc: %.4f",
                    task_idx, avg_forgetting,
                    glob_acc if glob_acc is not None else np.nanmean(task_accs))

        # TensorBoard
        if self.tb_writer is not None:
            self.tb_writer.add_scalar('metrics/avg_forgetting',
                                      avg_forgetting, task_idx)
            if glob_acc is not None:
                self.tb_writer.add_scalar('metrics/global_accuracy',
                                          glob_acc, task_idx)
            for i, f_i in enumerate(forgetting):
                self.tb_writer.add_scalar(f'forgetting/task_{i}', f_i, task_idx)

    def _compute_forgetting(self, current_task: int) -> np.ndarray:
        """
        Compute forgetting for all tasks seen before `current_task`.

        Forgetting for task i after learning task t:
            F_i = max_{j in [0..t-1]} A[j][i]  -  A[t][i]

        Returns
        -------
        np.ndarray of shape [current_task]
            Forgetting values for tasks 0..current_task-1.
            (Task current_task itself cannot have forgetting yet.)
        """
        if current_task == 0:
            return np.array([])

        forgetting = np.zeros(current_task)
        for i in range(current_task):
            # Best accuracy on task i achieved during tasks 0..current_task-1
            past_accs = self.accuracy_matrix[:current_task, i]
            past_accs = past_accs[~np.isnan(past_accs)]
            if len(past_accs) == 0:
                forgetting[i] = 0.0
            else:
                best_past = np.max(past_accs)
                current_acc = self.accuracy_matrix[current_task, i]
                # F_i = max(past) - current (positive = forgetting occurred)
                forgetting[i] = best_past - current_acc if not np.isnan(current_acc) else 0.0
        return forgetting

    def get_forgetting_per_task(self) -> np.ndarray:
        """
        Compute final forgetting for each task after all tasks are learned.

        Returns
        -------
        np.ndarray, shape [num_tasks]
        """
        T = self.num_tasks
        forgetting = np.zeros(T)
        for i in range(T):
            col = self.accuracy_matrix[:, i]
            valid = col[~np.isnan(col)]
            if len(valid) <= 1:
                forgetting[i] = 0.0
            else:
                # Best accuracy across all evaluations except the last
                best_past = np.max(valid[:-1])
                forgetting[i] = best_past - valid[-1]
        return forgetting

    def compute_backward_transfer(self) -> float:
        """
        Backward Transfer (BWT):
            BWT = (1/(T-1)) * Σ_{i=0}^{T-2} (A[T-1][i] - A[i][i])

        Negative BWT indicates forgetting; positive indicates synergy.
        """
        T = self.num_tasks
        if T <= 1:
            return 0.0
        bwt_sum = 0.0
        count = 0
        for i in range(T - 1):
            final_acc = self.accuracy_matrix[T - 1, i]
            diag_acc = self.accuracy_matrix[i, i]
            if not np.isnan(final_acc) and not np.isnan(diag_acc):
                bwt_sum += (final_acc - diag_acc)
                count += 1
        return bwt_sum / count if count > 0 else 0.0

    def compute_forward_transfer(self, baseline_accs: np.ndarray = None) -> float:
        """
        Forward Transfer (FWT):
            FWT = (1/(T-1)) * Σ_{i=1}^{T-1} (A[i-1][i] - b_i)

        Where b_i is the baseline accuracy on task i (default: 1/num_classes
        = random chance, which we approximate as 0 since we don't know
        num_classes per task here).

        Parameters
        ----------
        baseline_accs : np.ndarray, optional
            Per-task random-chance baseline. If None, assumed 0.
        """
        T = self.num_tasks
        if T <= 1:
            return 0.0
        if baseline_accs is None:
            baseline_accs = np.zeros(T)
        fwt_sum = 0.0
        count = 0
        for i in range(1, T):
            # A[i-1][i] = accuracy on task i *before* learning it
            pre_acc = self.accuracy_matrix[i - 1, i]
            if not np.isnan(pre_acc):
                fwt_sum += (pre_acc - baseline_accs[i])
                count += 1
        return fwt_sum / count if count > 0 else 0.0

    # ======================================================================
    #  2. COMMUNICATION COST
    # ======================================================================

    def record_communication_round(self, round_idx: int, model,
                                   num_clients: int = None):
        """
        Record communication cost for one federated aggregation round.

        Cost = num_params × bytes_per_param × 2 (upload + download) × num_clients

        Parameters
        ----------
        round_idx : int
            Global round index.
        model : PreciseModel
            The model being aggregated (used to count parameters).
        num_clients : int
            Number of participating clients this round.
        """
        if num_clients is None:
            num_clients = self.num_clients

        # Count total trainable parameters
        total_params = sum(p.numel() for p in model.parameters())

        # Bytes per parameter (float32 = 4 bytes)
        bytes_per_param = 4
        model_size_bytes = total_params * bytes_per_param

        # Each client uploads its model; server broadcasts aggregated model
        # Upload: num_clients × model_size
        # Download: num_clients × model_size
        upload_bytes = num_clients * model_size_bytes
        download_bytes = num_clients * model_size_bytes
        round_total_bytes = upload_bytes + download_bytes

        round_record = {
            'round_idx': round_idx,
            'num_params': total_params,
            'model_size_bytes': model_size_bytes,
            'model_size_mb': model_size_bytes / (1024 ** 2),
            'num_clients': num_clients,
            'upload_bytes': upload_bytes,
            'download_bytes': download_bytes,
            'round_total_bytes': round_total_bytes,
            'round_total_mb': round_total_bytes / (1024 ** 2),
            'upload_mb': upload_bytes / (1024 ** 2),
            'download_mb': download_bytes / (1024 ** 2),
        }
        self.comm_rounds.append(round_record)

        # TensorBoard
        if self.tb_writer is not None:
            cumulative_mb = sum(r['round_total_mb'] for r in self.comm_rounds)
            self.tb_writer.add_scalar('communication/round_total_mb',
                                      round_record['round_total_mb'], round_idx)
            self.tb_writer.add_scalar('communication/cumulative_mb',
                                      cumulative_mb, round_idx)

    def get_communication_summary(self) -> dict:
        """Return aggregate communication statistics."""
        if len(self.comm_rounds) == 0:
            return {}
        total_bytes = sum(r['round_total_bytes'] for r in self.comm_rounds)
        total_upload = sum(r['upload_bytes'] for r in self.comm_rounds)
        total_download = sum(r['download_bytes'] for r in self.comm_rounds)
        avg_per_round = total_bytes / len(self.comm_rounds)
        avg_per_client_per_round = avg_per_round / self.num_clients if self.num_clients > 0 else 0

        return {
            'total_rounds': len(self.comm_rounds),
            'total_params_per_model': self.comm_rounds[0]['num_params'],
            'model_size_mb': self.comm_rounds[0]['model_size_mb'],
            'total_upload_mb': total_upload / (1024 ** 2),
            'total_download_mb': total_download / (1024 ** 2),
            'total_communication_mb': total_bytes / (1024 ** 2),
            'avg_per_round_mb': avg_per_round / (1024 ** 2),
            'avg_per_client_per_round_mb': avg_per_client_per_round / (1024 ** 2),
        }

    # ======================================================================
    #  3. COSINE SIMILARITY (store results from CosineTracker)
    # ======================================================================

    def record_cosine_stats(self, task_idx: int, stats: dict):
        """
        Store cosine similarity statistics for a given task.

        Parameters
        ----------
        task_idx : int
        stats : dict
            Keys: mean, std, min, max, num_accepted, num_rejected, total
        """
        self.cosine_stats_per_task[task_idx] = stats

        if self.tb_writer is not None:
            for k, v in stats.items():
                if isinstance(v, (int, float)):
                    self.tb_writer.add_scalar(f'cosine/{k}', v, task_idx)

    # ======================================================================
    #  4. FAIRNESS ANALYSIS
    # ======================================================================

    def run_fairness_analysis(self, users):
        """
        Run fairness analysis on the final model predictions.

        Collects predictions and labels from all users' test data,
        assigns protected groups via class-label partitioning, and
        computes all fairness metrics.

        Parameters
        ----------
        users : list of UserPreciseFCL
        """
        from utils.fairness_metrics import (
            run_full_fairness_analysis,
            assign_groups_by_class,
        )

        all_preds = []
        all_labels = []

        for user in users:
            # Use the user's test_all_ which evaluates on all tasks seen
            # We only need predictions and labels
            try:
                task_accs, task_losses, task_samples, preds, labels = \
                    user.test_all_(personal=False, matrix=True)
                all_preds.extend(preds)
                all_labels.extend(labels)
            except Exception as e:
                logger.warning("Fairness: skipping user %s: %s", user.id, str(e))
                continue

        if len(all_preds) == 0:
            logger.warning("No predictions collected for fairness analysis")
            return

        preds_arr = np.array(all_preds)
        labels_arr = np.array(all_labels)

        # Assign protected groups based on class labels (even/odd split)
        groups = assign_groups_by_class(labels_arr, strategy='even_odd')

        # Run full fairness analysis
        self.fairness_results = run_full_fairness_analysis(
            preds_arr, labels_arr, groups
        )

        logger.info("[ResearchMetrics] Fairness analysis complete")

        # TensorBoard
        if self.tb_writer is not None:
            for metric_name, value in self.fairness_results.items():
                if isinstance(value, (int, float)):
                    self.tb_writer.add_scalar(f'fairness/{metric_name}', value, 0)

    # ======================================================================
    #  5. EXPORT
    # ======================================================================

    def save_all(self):
        """Save all metrics to CSV and JSON files."""
        self._save_forgetting_csv()
        self._save_communication_csv()
        self._save_cosine_csv()
        self._save_fairness_csv()
        self._save_summary_json()
        logger.info("[ResearchMetrics] All metrics saved to %s", self.output_dir)

    def _save_forgetting_csv(self):
        """Save forgetting metrics to forgetting_metrics.csv."""
        forgetting = self.get_forgetting_per_task()
        rows = []
        for i in range(self.num_tasks):
            row = {
                'task': i,
                'forgetting': forgetting[i],
            }
            # Add accuracy history for this task across all evaluation points
            for t in range(self.num_tasks):
                acc = self.accuracy_matrix[t, i]
                row[f'acc_after_task_{t}'] = acc if not np.isnan(acc) else ''
            rows.append(row)

        # Add summary row
        rows.append({
            'task': 'average',
            'forgetting': np.mean(forgetting),
        })

        df = pd.DataFrame(rows)
        path = os.path.join(self.output_dir, 'forgetting_metrics.csv')
        df.to_csv(path, index=False)
        logger.info("  → %s", path)

    def _save_communication_csv(self):
        """Save communication metrics to communication_metrics.csv."""
        if len(self.comm_rounds) == 0:
            return
        df = pd.DataFrame(self.comm_rounds)
        # Add cumulative column
        df['cumulative_mb'] = df['round_total_mb'].cumsum()
        path = os.path.join(self.output_dir, 'communication_metrics.csv')
        df.to_csv(path, index=False)
        logger.info("  → %s", path)

    def _save_cosine_csv(self):
        """Save cosine analysis to cosine_analysis.csv."""
        if len(self.cosine_stats_per_task) == 0:
            return
        rows = []
        for task_idx in sorted(self.cosine_stats_per_task.keys()):
            row = {'task': task_idx}
            row.update(self.cosine_stats_per_task[task_idx])
            rows.append(row)
        df = pd.DataFrame(rows)
        path = os.path.join(self.output_dir, 'cosine_analysis.csv')
        df.to_csv(path, index=False)
        logger.info("  → %s", path)

    def _save_fairness_csv(self):
        """Save fairness/bias metrics to bias_metrics.csv."""
        if len(self.fairness_results) == 0:
            return

        # Separate scalar metrics from per-group data
        scalar_metrics = {}
        group_data = {}
        for k, v in self.fairness_results.items():
            if isinstance(v, dict):
                group_data[k] = v
            else:
                scalar_metrics[k] = v

        # Save scalar metrics
        if scalar_metrics:
            df_scalar = pd.DataFrame([scalar_metrics])
            path = os.path.join(self.output_dir, 'bias_metrics.csv')
            df_scalar.to_csv(path, index=False)
            logger.info("  → %s", path)

        # Save per-group confusion matrices if available
        if 'group_0_metrics' in group_data or 'group_1_metrics' in group_data:
            rows = []
            for gkey in ['group_0_metrics', 'group_1_metrics']:
                if gkey in group_data:
                    row = {'group': gkey}
                    row.update(group_data[gkey])
                    rows.append(row)
            if rows:
                df_groups = pd.DataFrame(rows)
                path = os.path.join(self.output_dir, 'bias_metrics_per_group.csv')
                df_groups.to_csv(path, index=False)
                logger.info("  → %s", path)

    def _save_summary_json(self):
        """Save comprehensive summary to research_metrics_summary.json."""
        forgetting = self.get_forgetting_per_task()
        bwt = self.compute_backward_transfer()
        fwt = self.compute_forward_transfer()
        comm = self.get_communication_summary()

        summary = {
            'training_time_seconds': time.time() - self._start_time,
            'num_tasks': self.num_tasks,
            'num_clients': self.num_clients,
            'accuracy': {
                'final_average_accuracy': float(np.nanmean(
                    self.accuracy_matrix[self.num_tasks - 1, :]
                )) if not np.all(np.isnan(self.accuracy_matrix[-1])) else None,
                'accuracy_per_task_final': [
                    float(x) if not np.isnan(x) else None
                    for x in self.accuracy_matrix[self.num_tasks - 1, :]
                ],
                'global_acc_per_task': [float(x) for x in self.glob_acc_per_task],
            },
            'forgetting': {
                'per_task': [float(x) for x in forgetting],
                'average_forgetting': float(np.mean(forgetting)),
            },
            'transfer': {
                'backward_transfer_BWT': float(bwt),
                'forward_transfer_FWT': float(fwt),
            },
            'communication': comm,
            'cosine_similarity': {
                str(k): v for k, v in self.cosine_stats_per_task.items()
            },
            'fairness': {
                k: float(v) if isinstance(v, (int, float, np.floating)) else v
                for k, v in self.fairness_results.items()
                if not isinstance(v, dict)
            },
            'accuracy_matrix': self.accuracy_matrix.tolist(),
        }

        path = os.path.join(self.output_dir, 'research_metrics_summary.json')
        with open(path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        logger.info("  → %s", path)

    # ======================================================================
    #  6. PLOTS
    # ======================================================================

    def generate_plots(self):
        """Generate all research plots."""
        from utils.visualization import generate_all_plots
        try:
            generate_all_plots(self, self.plots_dir)
            logger.info("[ResearchMetrics] Plots saved to %s", self.plots_dir)
        except Exception as e:
            logger.warning("Plot generation failed: %s", str(e))

    # ======================================================================
    #  7. FINAL SUMMARY
    # ======================================================================

    def print_final_summary(self):
        """Print a comprehensive summary table at the end of training."""
        forgetting = self.get_forgetting_per_task()
        bwt = self.compute_backward_transfer()
        fwt = self.compute_forward_transfer()
        comm = self.get_communication_summary()

        final_acc = self.accuracy_matrix[self.num_tasks - 1, :]
        avg_acc = float(np.nanmean(final_acc))

        sep = "=" * 70
        logger.info("\n%s", sep)
        logger.info("          RESEARCH METRICS — FINAL SUMMARY")
        logger.info("%s", sep)

        # Accuracy
        logger.info("\n--- Accuracy ---")
        logger.info("  Final Average Accuracy:      %.4f %%", avg_acc * 100)
        for i in range(self.num_tasks):
            val = final_acc[i]
            logger.info("    Task %d:  %.4f %%", i,
                        val * 100 if not np.isnan(val) else 0.0)

        # Forgetting
        logger.info("\n--- Forgetting ---")
        logger.info("  Average Forgetting:          %.4f", np.mean(forgetting))
        for i, f_i in enumerate(forgetting):
            logger.info("    Task %d:  %.4f", i, f_i)

        # Transfer
        logger.info("\n--- Transfer Metrics ---")
        logger.info("  Backward Transfer (BWT):     %.4f", bwt)
        logger.info("  Forward Transfer  (FWT):     %.4f", fwt)

        # Communication
        logger.info("\n--- Communication Cost ---")
        if comm:
            logger.info("  Total Communication:         %.2f MB", comm['total_communication_mb'])
            logger.info("  Total Upload:                %.2f MB", comm['total_upload_mb'])
            logger.info("  Total Download:              %.2f MB", comm['total_download_mb'])
            logger.info("  Avg per Round:               %.2f MB", comm['avg_per_round_mb'])
            logger.info("  Avg per Client per Round:    %.2f MB", comm['avg_per_client_per_round_mb'])
            logger.info("  Model Size:                  %.2f MB (%d params)",
                        comm['model_size_mb'], comm['total_params_per_model'])
        else:
            logger.info("  (no communication data recorded)")

        # Fairness
        logger.info("\n--- Fairness Metrics ---")
        if self.fairness_results:
            for k, v in self.fairness_results.items():
                if isinstance(v, (int, float, np.floating)):
                    logger.info("  %-35s %.4f", k + ":", float(v))
        else:
            logger.info("  (no fairness data recorded)")

        # Cosine Similarity
        logger.info("\n--- Cosine Similarity Statistics ---")
        if self.cosine_stats_per_task:
            for task_idx in sorted(self.cosine_stats_per_task.keys()):
                stats = self.cosine_stats_per_task[task_idx]
                logger.info("  Task %d: mean=%.4f  std=%.4f  "
                            "accepted=%d  rejected=%d",
                            task_idx,
                            stats.get('mean', 0), stats.get('std', 0),
                            stats.get('num_accepted', 0),
                            stats.get('num_rejected', 0))
        else:
            logger.info("  (no cosine data recorded)")

        logger.info("\n%s", sep)
        logger.info("  Training time: %.1f seconds", time.time() - self._start_time)
        logger.info("%s\n", sep)

        # Close TensorBoard writer
        if self.tb_writer is not None:
            self.tb_writer.close()
