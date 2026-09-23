# MGA local research fork; includes modifications relative to upstream Plato/KNOT.
# Distributed under Apache-2.0; see LICENSE, NOTICE and MODIFICATIONS.md.
"""
A customized server for Knot, a clustered aggregation mechanism designed for
federated unlearning.
"""

import copy
import json
import logging
import math
import os
import random
from collections import deque

import numpy
import torch
import torch.nn.functional as F
from cvxopt import matrix
from numpy.linalg import norm

from plato.config import Config
from plato.utils import fonts

import fedunlearning_server
import solver
from mga_protocol import MGAPipelineMixin


class Server(MGAPipelineMixin, fedunlearning_server.Server):
    """
    A federated unlearning server that implements the federated unlearning clustering algorithm.

    The work pipeline of the server is as follows:

    1. The server divides the total number of clients randomly into a number of clusters.
    2. The clients carry out the training normally.
    3. Model updates from the clients will be aggregated according to the clustering assignments.
    4. Send the model corresponding to each cluster to its corresponding clients.
    5. Perform global aggregation (aggregate from all clusters) after convergence or when the
       target round has reached.
    """

    def __init__(self, model=None, datasource=None, algorithm=None, trainer=None):
        super().__init__(
            model=model, datasource=datasource, algorithm=algorithm, trainer=trainer
        )

        # A dictionary that maps client IDs to the cluster IDs
        self.num_clusters = 0
        self.clusters = {}

        # If HuggingFace models are used, set the initial accuracy to a very large value
        if hasattr(Config().trainer, "target_perplexity"):
            self.accuracy = 1000

        # A dictionary that maps the cluster ID to a boolean value that shows whether
        # this cluster is currently going through the retraining phase
        self.clustered_retraining = {}

        # A dictionary that maps cluster IDs to the accuracy belonging to each cluster
        # This accuracy is computed based on the test dataset on the server side, i.e.,
        # each cluster will test its updated model after aggregation) on the server's
        # test set to obtain its accuracy.
        self.clustered_test_accuracy = {}

        # The recent history of global accuracies
        self.recent_global_accuracies = None
        self.recent_history_size = None

        # A dictionary that maps the cluster ID to the earliest round that retraining
        # must start from when entering the retraining phase
        self.earliest_round = {}

        # A dictionary that maps the cluster ID to the round when the retraining process
        # should roll back to, which is the earliest_round above minus 1
        self.rollback_round = {}

        # A dictionary that maps cluster IDs to the updates belonging to each cluster
        self.clustered_updates = {}

        # A pre-trained server model generated before the unlearning, for cos similarity
        self.pretrained_server_model = None

        self.server_aggregation_count = 0
        self.client_aggregation_counts = {}
        self._last_aggregated_client_ids = []

        # A dictionary that maps client ids and its cos similarity compared with
        # pre-trained server model
        self.clients_similarity = {}

        # Whether we are using random clustering or optimized clustering
        self.initialize_optimization = False

        # Initialize basic metrics required by clusters, such as accuracites
        self._init_cluster_states()

        # Initialize the function that clustering clients by solver or randomly
        self._clustering_clients()
        self._init_mga_pipeline()

    def init_trainer(self) -> None:
        """Load the trainer and initialize the dictionary that maps cluster IDs to client IDs."""
        super().init_trainer()

        initial_model_state_path = getattr(
            Config().server, "initial_model_state_path", None
        )
        if initial_model_state_path:
            initial_state = torch.load(initial_model_state_path, map_location="cpu")
            self.trainer.model.load_state_dict(initial_state, strict=True)
            logging.info(
                "[%s] Loaded common scratch initialization from %s.",
                self,
                initial_model_state_path,
            )

        self.algorithm.init_clusters(self.clusters)
        self._true_retrain_initial_state = copy.deepcopy(
            self.trainer.model.state_dict()
        )
        self._true_retrain_reset_done = False

    def _maybe_reset_true_retrain(self):
        """Reset the complete server state before the first retained-data retrain round."""
        if not bool(getattr(Config().trainer, "true_retrain_reset", False)):
            return
        if self._true_retrain_reset_done:
            return
        if self.current_round != int(Config().clients.data_deletion_round):
            return

        initial_state = copy.deepcopy(self._true_retrain_initial_state)
        self.trainer.model.load_state_dict(initial_state, strict=True)
        self.algorithm.models = {}
        self.algorithm.init_clusters(self.clusters)
        for cluster_id in range(self.num_clusters):
            self.algorithm.load_weights(
                copy.deepcopy(initial_state), cluster_id=cluster_id
            )

        self.clustered_updates = {}
        if hasattr(self, "updates"):
            self.updates = []
        self.server_aggregation_count = 0
        self.client_aggregation_counts = {}
        self._last_aggregated_client_ids = []
        self.pretrained_server_model = copy.deepcopy(self.trainer.model)
        self.clustered_retraining = {
            cluster_id: True for cluster_id in range(self.num_clusters)
        }
        self.rollback_round = {
            cluster_id: 0 for cluster_id in range(self.num_clusters)
        }
        self.earliest_round = {
            cluster_id: 1 for cluster_id in range(self.num_clusters)
        }
        self.clustered_test_accuracy = {
            cluster_id: 0 for cluster_id in range(self.num_clusters)
        }
        if self.recent_global_accuracies is not None:
            self.recent_global_accuracies.clear()
        self.accuracy = 0
        self._true_retrain_reset_done = True
        logging.info(
            "[%s] TRUE_RETRAIN_RESET round=%s: global and %s cluster models "
            "reinitialized; asynchronous aggregation history cleared.",
            self,
            self.current_round,
            len(self.algorithm.models),
        )

    def choose_clients(self, clients_pool, clients_count):
        """
        Choose a subset of clients to participate in each round.
        When do_optimized_clustering is true, the first and second round is to
        extract training time and cos similarity for all clients;
        after and at second round, training process resume.
        """
        self._maybe_reset_true_retrain()
        if getattr(self, "_mga_enabled", False):
            if self._mga_phase == "unlearn":
                return self._mga_choose_targets(clients_pool, clients_count)
            if self._mga_phase == "recover":
                targets = set(self._mga_request.targets)
                clients_pool = [i for i in clients_pool if i not in targets]
                clients_count = min(clients_count, len(clients_pool))
                if clients_count < 1:
                    raise RuntimeError("No retained client is available for recovery.")
                random.setstate(self.prng_state)
                selected = random.sample(clients_pool, clients_count)
                self.prng_state = random.getstate()
                self._mga_event("recovery_selected", clients=selected)
                return selected
        assert clients_count <= len(clients_pool)
        random.setstate(self.prng_state)

        if (
            hasattr(Config().server, "do_optimized_clustering")
            and Config().server.do_optimized_clustering
        ):
            if self.current_round <= 2 and self.initialize_optimization:
                clients_count = len(clients_pool)
                self.minimum_clients = clients_count
            elif self.current_round == 3:
                self.minimum_clients = Config.server.minimum_clients_aggregated
                self.clients_per_round = Config().clients.per_round
                clients_count = self.clients_per_round

        # Select clients randomly
        if (
            hasattr(Config().server, "exclude_delete_clients_after_deletion")
            and Config().server.exclude_delete_clients_after_deletion
            and self.current_round >= Config().clients.data_deletion_round
        ):
            delete_clients = set(Config().clients.clients_requesting_deletion)
            retained_clients_pool = [
                client_id
                for client_id in clients_pool
                if client_id not in delete_clients
            ]
            if len(retained_clients_pool) >= clients_count:
                clients_pool = retained_clients_pool
                logging.info(
                    "[%s] Excluding deletion-requesting clients after round %s.",
                    self,
                    Config().clients.data_deletion_round,
                )

        controlled_selection = bool(
            getattr(Config().server, "controlled_client_selection", False)
        )
        if controlled_selection:
            available_clients = sorted(set(clients_pool))
            selection_seed = int(
                getattr(Config().server, "controlled_selection_seed", 20260806)
            )
            selection_rng = random.Random(
                selection_seed + int(self.current_round) * 10007
            )
            forced_clients = []
            deletion_round = int(Config().clients.data_deletion_round)
            unlearning_window = int(
                getattr(Config().trainer, "unlearning_window", 0) or 0
            )
            in_unlearning_window = (
                deletion_round <= self.current_round
                and (
                    unlearning_window <= 0
                    or self.current_round < deletion_round + unlearning_window
                )
            )
            if in_unlearning_window:
                deletion_clients = [
                    client_id
                    for client_id in sorted(
                        Config().clients.clients_requesting_deletion
                    )
                    if client_id in available_clients
                ]
                quota = min(
                    int(
                        getattr(
                            Config().server,
                            "controlled_delete_clients_per_round",
                            2,
                        )
                    ),
                    len(deletion_clients),
                    clients_count,
                )
                if quota > 0:
                    offset = (self.current_round - deletion_round) * quota
                    forced_clients = [
                        deletion_clients[(offset + index) % len(deletion_clients)]
                        for index in range(quota)
                    ]
            remaining_clients = [
                client_id
                for client_id in available_clients
                if client_id not in forced_clients
            ]
            remaining_count = clients_count - len(forced_clients)
            selected_clients = forced_clients + selection_rng.sample(
                remaining_clients, remaining_count
            )
            logging.info(
                "[%s] Controlled selection forced=%s seed=%s.",
                self,
                forced_clients,
                selection_seed + int(self.current_round) * 10007,
            )
        else:
            selected_clients = random.sample(clients_pool, clients_count)

        self.prng_state = random.getstate()
        logging.info("[%s] Selected clients: %s", self, selected_clients)
        return selected_clients

    def clients_selected(self, selected_clients):
        """Remember first participation for grouped and ungrouped rollback."""
        if not self.initialize_optimization:
            for client_id in selected_clients:
                if client_id not in self.round_first_selected:
                    self.round_first_selected[client_id] = self.current_round

    async def aggregate_weights(self, updates, baseline_weights, weights_received):
        """
        Aggregate the reported weight updates from the selected clients,
        according to each client's clustering assignment, using the
        federated averaging algorithm.

        Only clients belonging to the same cluster will be aggregated.
        """
        self._last_aggregated_client_ids = []
        if getattr(self, "_mga_enabled", False):
            if self._mga_reverse_round:
                return self._mga_accept_returns(updates)
            if self._mga_phase == "recover":
                for update in updates:
                    if (update.client_id in self._mga_request.targets
                            or getattr(update.report, "mga_phase", None) != "recover"
                            or getattr(update.report, "mga_request_id", None) != self._mga_request.request_id):
                        raise RuntimeError("Stale, target, or untagged update reached MGA recovery.")
        if (
            self.current_round == 1
            and (
                hasattr(Config().server, "do_optimized_clustering")
                and Config().server.do_optimized_clustering
            )
            and self.initialize_optimization
        ):
            # Perform normal server aggregation by first computing weight deltas
            deltas_received = self.algorithm.compute_weight_deltas(
                baseline_weights, weights_received, cluster_id=None
            )
            # and then run a framework-agnostic server aggregation algorithm, such as
            # the federated averaging algorithm
            logging.info("[%s] Aggregating model weight deltas.", self)
            deltas = await self.aggregate_deltas(self.updates, deltas_received)
            # Updates the existing model weights from the provided deltas
            updated_weights = self.algorithm.update_weights(deltas)
            self._last_aggregated_client_ids = [update.client_id for update in updates]
            return updated_weights

        if (
            self.current_round == 2
            and hasattr(Config().server, "do_optimized_clustering")
            and Config().server.do_optimized_clustering
            and self.initialize_optimization
        ):
            # Build the cluster map, then continue aggregating this round's
            # scratch updates instead of testing the untrained baseline.
            self._optimize_clustering(updates)
            for cluster_id in self.algorithm.models:
                self.algorithm.load_weights(baseline_weights, cluster_id=cluster_id)

        if (
            hasattr(Config().server, "retained_fedavg_after_deletion")
            and Config().server.retained_fedavg_after_deletion
            and self.current_round >= Config().clients.data_deletion_round
        ):
            delete_clients = set(Config().clients.clients_requesting_deletion)
            retained_updates = [
                update for update in updates if update.client_id not in delete_clients
            ]
            if retained_updates:
                retained_weights = [update.payload for update in retained_updates]
                deltas_received = self.algorithm.compute_weight_deltas(
                    baseline_weights, retained_weights, cluster_id=None
                )
                deltas = await self.aggregate_deltas(retained_updates, deltas_received)
                updated_weights = self.algorithm.update_weights(deltas)
                self.algorithm.load_weights(updated_weights)
                for cluster_id in self.algorithm.models:
                    self.algorithm.load_weights(updated_weights, cluster_id=cluster_id)
                self._last_aggregated_client_ids = [
                    update.client_id for update in retained_updates
                ]
                self._use_retained_fedavg_global = True
                logging.info(
                    "[%s] Retained FedAvg after deletion used %s clients: %s.",
                    self,
                    len(retained_updates),
                    self._last_aggregated_client_ids,
                )
                return updated_weights

        # Keep the pre-unlearning baseline identical across all scratch variants.
        if (
            self.current_round < Config().clients.data_deletion_round
            and hasattr(Config().server, "do_optimized_clustering")
            and Config().server.do_optimized_clustering
        ):
            deltas_received = self.algorithm.compute_weight_deltas(
                baseline_weights, weights_received, cluster_id=None
            )
            deltas = await self.aggregate_deltas(self.updates, deltas_received)
            updated_weights = self.algorithm.update_weights(deltas)
            self.algorithm.load_weights(updated_weights)
            for cluster_id in self.algorithm.models:
                self.algorithm.load_weights(updated_weights, cluster_id=cluster_id)
            self._last_aggregated_client_ids = [update.client_id for update in updates]
            return updated_weights

        self.clustered_updates = {}
        aggregated_client_ids = set()
        for client_update in updates:
            client_id = client_update.client_id

            cluster_id = self.clusters.get(client_id)
            if cluster_id is None:
                continue

            if self.clustered_retraining.get(cluster_id, False) is True:
                if (
                    abs(client_update.staleness)
                    <= self.current_round - Config().clients.data_deletion_round - 1
                ):
                    if cluster_id in self.clustered_updates:
                        self.clustered_updates[cluster_id].append(client_update)
                    else:
                        self.clustered_updates[cluster_id] = [client_update]
                    aggregated_client_ids.add(client_id)

            else:
                if cluster_id in self.clustered_updates:
                    self.clustered_updates[cluster_id].append(client_update)
                else:
                    self.clustered_updates[cluster_id] = [client_update]
                aggregated_client_ids.add(client_id)

            for cluster_id, update in self.clustered_updates.items():
                if len(update) != 0:
                    # Perform server aggregation within a cluster (with cluster_id)
                    weights_received = [_update.payload for _update in update]

                    deltas_received = self.algorithm.compute_weight_deltas(
                        baseline_weights, weights_received, cluster_id=cluster_id
                    )
                    deltas = await self.aggregate_deltas(update, deltas_received)
                    updated_weights = self.algorithm.update_weights(
                        deltas, cluster_id=cluster_id
                    )

                    self.algorithm.load_weights(updated_weights, cluster_id=cluster_id)

        self._last_aggregated_client_ids = sorted(aggregated_client_ids)
        return baseline_weights

    def weights_aggregated(self, updates):
        """Method called after the updated weights have been aggregated."""
        if getattr(self, "_mga_enabled", False) and self._mga_reverse_round:
            # A target batch must neither increment training participation counts
            # nor be overwritten by legacy cluster/global model selection.
            return
        aggregated_client_ids = getattr(self, "_last_aggregated_client_ids", [])
        if aggregated_client_ids:
            self.server_aggregation_count += 1
            for client_id in aggregated_client_ids:
                self.client_aggregation_counts[client_id] = (
                    self.client_aggregation_counts.get(client_id, 0) + 1
                )

            aggregation_counts = numpy.zeros(
                int(Config().clients.total_clients) + 1, dtype=numpy.float64
            )
            for counted_client_id, count in self.client_aggregation_counts.items():
                if 0 <= int(counted_client_id) < aggregation_counts.shape[0]:
                    aggregation_counts[int(counted_client_id)] = float(count)
            temporary_counts_path = "client_aggregation_counts.npy.tmp"
            with open(temporary_counts_path, "wb") as counts_file:
                numpy.save(counts_file, aggregation_counts)
            os.replace(temporary_counts_path, "client_aggregation_counts.npy")

            logging.info(
                "[%s] Aggregation count=%s; client aggregation counts updated for %s.",
                self,
                self.server_aggregation_count,
                aggregated_client_ids,
            )

        self._mga_capture_retained(updates)

        # Testing the updated clustered model directly at the server
        if (
            hasattr(Config().server, "do_clustered_test")
            and Config().server.do_clustered_test
            and not self.initialize_optimization
        ):
            # First, obtain the set of cluster IDs that have been aggregated
            updated_cluster_ids = set()

            for update in self.updates:
                cluster_id = self.clusters.get(update.client_id)
                if cluster_id is None:
                    continue

                updated_cluster_ids.add(cluster_id)

            # updated_cluster_ids = {
            #     self.clusters[update.client_id] for update in self.updates
            # }

            test_accuracy_per_cluster = self.trainer.server_clustered_test(
                self.testset,
                self.testset_sampler,
                clustered_models=self.algorithm.models,
                updated_cluster_ids=updated_cluster_ids,
            )

            # Second, update the test accuracy for clusters that have just been tested
            self.clustered_test_accuracy.update(test_accuracy_per_cluster)

        if hasattr(Config().server, "do_test") and Config().server.do_test:
            if getattr(self, "_use_retained_fedavg_global", False):
                self._use_retained_fedavg_global = False
            else:
                # Retrieve the model from the cluster with the highest accuracy
                self.trainer.model.load_state_dict(self._aggregate_models(), strict=True)

    def clients_processed(self):
        """Determining the rollback round and roll back to that round, if retraining is needed
        for each of the clusters."""
        # If data_deletion_round equals to the current round at server for the first time,
        # and the clients requesting retraining has been selected before, the retraining
        # phase starts.

        if getattr(self, "_mga_enabled", False) and self._mga_phase != "train":
            # The explicit request barrier replaces the old per-cluster rollback.
            return

        # Make sure that the model at round 0 is loaded at round 3, if we
        # have activated the optimized clustering algorithm
        if (
            hasattr(Config().server, "do_optimized_clustering")
            and Config().server.do_optimized_clustering
            and self.current_round == 2
        ):
            for cluster_id in range(self.num_clusters):
                # Loading the saved model on the server for starting the retraining phase
                checkpoint_path = Config.params["checkpoint_path"]

                model_name = (
                    Config().trainer.model_name
                    if hasattr(Config().trainer, "model_name")
                    else "custom"
                )

                # Loading the model from round 0
                rollback_round = 0
                filename = f"checkpoint_{model_name}_{rollback_round}_{cluster_id}.pth"

                self._load_model(cluster_id, filename, checkpoint_path)

                logging.info(
                    fonts.colourize(
                        f"[Server #{os.getpid()}] Cluster #{cluster_id}: Round #0 "
                        f"model reloaded from {filename}.",
                        colour="green",
                    )
                )

        if not self.initialize_optimization:
            self.earliest_round = {
                cluster_id: self.current_round
                for cluster_id in range(self.num_clusters)
            }

        data_deletion_round = Config().clients.data_deletion_round
        clients_to_delete = Config().clients.clients_requesting_deletion

        for cluster_id in range(self.num_clusters):
            if (
                (self.current_round == data_deletion_round)
                and not self.clustered_retraining[cluster_id]
                and not self.initialize_optimization
            ):
                for client_id, init_round in self.round_first_selected.items():
                    mapped_cluster_id = self.clusters.get(client_id)
                    if (
                        client_id in clients_to_delete
                        and mapped_cluster_id is not None
                        and mapped_cluster_id == cluster_id
                    ):
                        self.clustered_retraining[cluster_id] = True

                        if self.earliest_round[cluster_id] > init_round:
                            self.earliest_round[cluster_id] = init_round

                        self.rollback_round[cluster_id] = (
                            self.earliest_round[cluster_id] - 1
                        )
                        logging.info(
                            fonts.colourize(
                                f"[{self}] Data deleted. Retraining cluster #{cluster_id} "
                                f"from the states after round #{self.rollback_round[cluster_id]}.",
                                colour="green",
                            )
                        )

                if (
                    cluster_id in self.rollback_round
                    and hasattr(Config().server, "skip_cluster_rollback_after_deletion")
                    and Config().server.skip_cluster_rollback_after_deletion
                ):
                    logging.info(
                        "[Server #%d] Cluster #%s rollback skipped; continuing "
                        "from the current model after deletion.",
                        os.getpid(),
                        cluster_id,
                    )
                elif cluster_id in self.rollback_round:
                    # Loading the saved model on the server for starting the retraining phase
                    checkpoint_path = Config.params["checkpoint_path"]

                    model_name = (
                        Config().trainer.model_name
                        if hasattr(Config().trainer, "model_name")
                        else "custom"
                    )

                    # When the current_round matches the data deletion round, load the model from
                    # `rollback_round`.
                    
                    rollback_round = self.rollback_round[cluster_id]
                    #rollback_round = 0
                    filename = (
                        f"checkpoint_{model_name}_{rollback_round}_{cluster_id}.pth"
                    )

                    self._load_model(cluster_id, filename, checkpoint_path)

                    logging.info(
                        "[Server #%d] Model in cluster #%s's retraining phase loaded from %s.",
                        os.getpid(),
                        cluster_id,
                        filename,
                    )

        if self.current_round >= 2:
            self.initialize_optimization = False

    def customize_server_response(self, server_response: dict, client_id) -> dict:
        """Returns a customrized server response with any additional information."""
        server_response = super().customize_server_response(
            server_response, client_id=client_id
        )

        if getattr(self, "_mga_enabled", False):
            return self._mga_response(server_response, client_id)

        cluster_id = self.clusters.get(client_id)
        if cluster_id is None:
            return server_response

        deletion_round = int(Config().clients.data_deletion_round)
        deletion_clients = set(Config().clients.clients_requesting_deletion)
        unlearning_window = int(getattr(Config().trainer, "unlearning_window", 1))
        if client_id in deletion_clients and self.current_round >= deletion_round:
            knot_retrain = bool(getattr(Config().trainer, "knot_retrain", False))
            in_unlearning_window = self.current_round < deletion_round + unlearning_window
            server_response["sampler_mode"] = (
                "retain" if knot_retrain or not in_unlearning_window else "forget"
            )

        if (
            hasattr(Config().server, "retained_recovery_epochs_after_deletion")
            and self.current_round >= Config().clients.data_deletion_round
            and client_id not in set(Config().clients.clients_requesting_deletion)
        ):
            server_response["local_epochs"] = int(
                Config().server.retained_recovery_epochs_after_deletion
            )

        fixed_unlearning_epochs = getattr(
            Config().trainer, "fixed_unlearning_epochs", None
        )
        if fixed_unlearning_epochs is None:
            client_aggregation_count = self.client_aggregation_counts.get(client_id, 0)
            dynamic_e_min = int(getattr(Config().trainer, "dynamic_e_min", 1))
            dynamic_e_max = int(getattr(Config().trainer, "dynamic_e_max", 5))
            dynamic_e_alpha = float(
                getattr(Config().trainer, "dynamic_e_alpha", 1.0)
            )
            dynamic_e = min(
                dynamic_e_max,
                max(
                    dynamic_e_min,
                    dynamic_e_min
                    + math.floor(
                        dynamic_e_alpha * math.log2(1 + client_aggregation_count)
                    ),
                ),
            )
            server_response["dynamic_e_alpha"] = dynamic_e_alpha
            server_response["dynamic_e_formula"] = "bounded_log2_client_exposure"
        else:
            dynamic_e = int(fixed_unlearning_epochs)
            server_response["fixed_e_ablation"] = True
        server_response["unlearning_epochs"] = dynamic_e
        server_response["server_aggregation_count"] = self.server_aggregation_count
        server_response["client_aggregation_count"] = self.client_aggregation_counts.get(
            client_id, 0
        )

        if (
            cluster_id in self.clustered_retraining
            and self.clustered_retraining[cluster_id]
        ):
            # Each cluster has its own round number that it needs to roll back to during the
            # retraining phase, which is the earliest round among clients in this cluster, minus 1
            server_response["rollback_round"] = self.rollback_round[cluster_id]

        return server_response

    def get_logged_items(self) -> dict:
        """Get items to be logged by the LogProgressCallback class in a .csv file."""
        logged_items = super().get_logged_items()

        clusters_accuracy = [
            self.clustered_test_accuracy[cluster_id]
            for cluster_id in range(self.num_clusters)
        ]

        clusters_accuracy = "; ".join([str(acc) for acc in clusters_accuracy])

        logged_items["clusters_accuracy"] = clusters_accuracy

        return logged_items

    def save_to_checkpoint(self) -> None:
        """Save a checkpoint for resuming the training session."""
        super().save_to_checkpoint()

        for cluster_id in range(self.num_clusters):
            checkpoint_path = Config.params["checkpoint_path"]

            model_name = (
                Config().trainer.model_name
                if hasattr(Config().trainer, "model_name")
                else "custom"
            )
            filename = f"checkpoint_{model_name}_{self.current_round}_{cluster_id}.pth"
            logging.info(
                "[%s] Saving the checkpoint for cluster #%s to %s/%s.",
                self,
                cluster_id,
                checkpoint_path,
                filename,
            )
            self._save_model(cluster_id, filename, checkpoint_path)

    async def wrap_up(self) -> None:
        """Wrapping up when each round of training is done."""
        self.save_to_checkpoint()
        if getattr(self, "_mga_enabled", False) and self._mga_phase in ("train", "drain", "unlearn"):
            # Do not let early utility convergence silently skip the request.
            if self.current_round >= int(Config().trainer.rounds):
                await self._close()  # fails closed if the request is incomplete
            return

        # Break the loop when the target accuracy is achieved
        target_accuracy = None
        target_perplexity = None

        force_stop_at_rounds = (
            hasattr(Config().trainer, "force_stop_at_rounds")
            and Config().trainer.force_stop_at_rounds
        )
        should_stop = (
            self.current_round >= Config().trainer.rounds
            if force_stop_at_rounds
            else (
                self._did_stablize()
                or self.current_round >= Config().trainer.rounds
                and max(self.clustered_retraining.values())
            )
        )
        if should_stop:
            logging.info(
                fonts.colourize(
                    f"[{self}] Terminating at round {self.current_round} out of "
                    f"{Config().trainer.rounds}.",
                    colour="green",
                )
            )

            if self.current_round >= Config().trainer.rounds:
                logging.info("Target number of training rounds reached.")
                await self._close()

            if hasattr(Config().trainer, "target_accuracy"):
                target_accuracy = Config().trainer.target_accuracy
            elif hasattr(Config().trainer, "target_perplexity"):
                target_perplexity = Config().trainer.target_perplexity

            if (not force_stop_at_rounds) and target_accuracy and self.accuracy >= target_accuracy:
                logging.info(
                    "[%s] Target accuracy and standard deviation reached.", self
                )
                await self._close()

            if (not force_stop_at_rounds) and target_perplexity and self.accuracy <= target_perplexity:
                logging.info(
                    "[%s] Target perplexity and standard deviation reached.", self
                )
                await self._close()

    def _load_model(self, cluster_id, filename=None, location=None):
        """Loading pre-trained model weights before retraining from a file."""
        model_path = Config().params["model_path"] if location is None else location
        model_name = Config().trainer.model_name

        if filename is not None:
            model_path = f"{model_path}/{filename}"
        else:
            model_path = f"{model_path}/{model_name}.pth"

        logging.info("[Server #%d] Loading a model from %s.", os.getpid(), model_path)

        if cluster_id in self.algorithm.models:
            model = self.algorithm.models[cluster_id]
        else:
            model = self.trainer.model

        model.load_state_dict(torch.load(model_path), strict=True)

    def _save_model(self, cluster_id, filename=None, location=None):
        """Saving the model to a file."""
        model_path = Config().params["model_path"] if location is None else location
        model_name = Config().trainer.model_name

        try:
            if not os.path.exists(model_path):
                os.makedirs(model_path)
        except FileExistsError:
            pass

        if filename is not None:
            model_path = f"{model_path}/{filename}"
        else:
            model_path = f"{model_path}/{model_name}.pth"

        if cluster_id in self.algorithm.models:
            model = self.algorithm.models[cluster_id]
        else:
            model = self.trainer.model
        torch.save(model.state_dict(), model_path)

        logging.info("[Server #%d] Model saved to %s.", os.getpid(), model_path)

    def _init_cluster_states(self):
        """Initialize the basic dictionaries of the clusters and clients."""
        self.num_clusters = Config().server.clusters

        if hasattr(Config().server, "do_optimized_clustering"):
            self.initialize_optimization = Config().server.do_optimized_clustering
        else:
            self.initialize_optimization = False

        # Initializing the dictionary of boolean values, one per cluster, to record whether
        # we have already entered the retraining phase
        self.clustered_retraining = {
            cluster_id: False for cluster_id in range(self.num_clusters)
        }

        # Initializing the dictionary of float numbers, one per cluster, to record the
        # test accuracy of each cluster
        self.clustered_test_accuracy = {
            cluster_id: 0 for cluster_id in range(self.num_clusters)
        }

        # Initializing the dictionary of cos similarity between pre-trained model and
        # clients' first time updates
        self.clients_similarity = {
            client_id: None for client_id in range(1, self.total_clients + 1)
        }

        # recent_history_size denotes the length of accuracy recording
        # Default: each cluster only records latest 5 test accuracy
        if hasattr(Config().server, "window_size"):
            self.recent_history_size = Config().server.window_size
        else:
            self.recent_history_size = 5

        # Initializing a record of global test accuracies
        self.recent_global_accuracies = deque([], maxlen=self.recent_history_size)

    def _clustering_clients(self):
        """
        Randomly divide clients by their client ids into several clusters, and aggregation
        in clusters. As the baseline of optimization clustering.
        """

        total_clients = Config().clients.total_clients
        self.num_clusters = Config().server.clusters

        # The client IDs range from 1 to total_clients, and they are to be distributed to
        # the clusters, ranging from 0 to (num_clusters - 1)
        random.seed(1)
        if (
            hasattr(Config().server, "do_optimized_clustering")
            and Config().server.do_optimized_clustering
        ):
            # initial self.clusters dictionaty
            for client_id in range(1, total_clients + 1):
                self.clusters[client_id] = None
        else:
            # randomly cluster
            for client_id in range(1, total_clients + 1):
                cluster_id = int(random.random() * self.num_clusters)
                self.clusters[client_id] = cluster_id
            print("clusters: ", self.clusters)

    def _aggregate_models(self):
        """Aggregate all cluster models by their effective training-data mass."""
        active_client_counts = {
            cluster_id: 0 for cluster_id in self.algorithm.models
        }
        for client_id, cluster_id in self.clusters.items():
            if cluster_id not in active_client_counts:
                continue
            active_client_counts[cluster_id] += 1

        partition_size = float(getattr(Config().data, "partition_size", 1) or 1)
        cluster_weights = {
            cluster_id: client_count * partition_size
            for cluster_id, client_count in active_client_counts.items()
            if client_count > 0
        }
        total_weight = sum(cluster_weights.values())
        if total_weight <= 0:
            return self.trainer.model.state_dict()

        aggregated_model = {}
        model_states = {
            cluster_id: model.state_dict()
            for cluster_id, model in self.algorithm.models.items()
            if cluster_id in cluster_weights
        }
        for name, template in self.trainer.model.state_dict().items():
            values = [
                (model_states[cluster_id][name].to(Config().device()), weight)
                for cluster_id, weight in cluster_weights.items()
            ]
            if torch.is_floating_point(template) or torch.is_complex(template):
                aggregated_value = torch.zeros_like(template, device=Config().device())
                for value, weight in values:
                    aggregated_value += value * (weight / total_weight)
            else:
                aggregated_value = torch.zeros_like(
                    template, dtype=torch.float64, device=Config().device()
                )
                for value, weight in values:
                    aggregated_value += value.to(torch.float64) * (weight / total_weight)
                aggregated_value = aggregated_value.round().to(template.dtype)
            aggregated_model[name] = aggregated_value
        return aggregated_model

    def _did_stablize(self):
        """
        Whether the training process should be terminated based on:
            - The global accuracy corresponds to the highest per-cluster test accuracy.
            - The standard deviation of highest per-cluster test accuracies has
              reached a particular target.
        """

        # We first need to exceed the target accuracy for all the clusters
        if hasattr(Config().trainer, "target_accuracy"):
            target_accuracy = Config().trainer.target_accuracy

            # Update the histories of recent global test accuracies
            global_acc = max(self.clustered_test_accuracy.values())
            self.recent_global_accuracies.append(global_acc)

            # We then need to compute the standard deviation of latest
            # self.recent_history_size global accuracies, and make sure that it
            # is lower than the target_accuracy_std
            standard_deviation = numpy.std(list(self.recent_global_accuracies))
            target_accuracy_std = Config().trainer.target_accuracy_std
            logging.info(
                "[%s] Standard deviation of recent global accuracies: %.2f.",
                self,
                standard_deviation,
            )

            if len(self.recent_global_accuracies) >= self.recent_history_size:
                return (
                    global_acc > target_accuracy
                    and standard_deviation < target_accuracy_std
                )
            else:
                return False

        if hasattr(Config().trainer, "target_perplexity"):
            target_perplexity = Config().trainer.target_perplexity

            global_acc = min(self.clustered_test_accuracy.values())
            # Update the histories of recent global test accuracies
            self.recent_global_accuracies.append(global_acc)

            # We then need to compute the standard deviation of latest
            # self.recent_history_size global accuracies, and make sure that it
            # is lower than the target_accuracy_std
            standard_deviation = numpy.std(list(self.recent_global_accuracies))

            target_perplexity_std = Config().trainer.target_perplexity_std

            if len(self.recent_global_accuracies) >= self.recent_history_size:
                return (
                    0 < global_acc < target_perplexity
                    and standard_deviation < target_perplexity_std
                )
            else:
                return False

    def _extract_training_times(self, updates):
        """Extract the training time from the report in client updates."""
        # Initialize a dictionary that maps client_ids to its training times
        client_training_times = {
            client_id: 0 for client_id in range(1, Config().clients.total_clients + 1)
        }

        for update in updates:
            if client_training_times[update.client_id] != 0:
                continue

            client_training_times[update.client_id] = update.report.training_time

        if bool(
            getattr(Config().server, "controlled_training_times", False)
        ):
            client_training_times = {
                client_id: 1.0 + ((client_id * 37) % 100) / 100.0
                for client_id in range(1, Config().clients.total_clients + 1)
            }
            logging.info(
                "[%s] Using deterministic client training-time anchors.", self
            )

        return client_training_times

    def _cosine_similarity(self, updates):
        """Compute the cosine similarity of the received updates and the difference
        between the first round model - initial model and clients' updates."""

        model_name = (
            Config().trainer.model_name
            if hasattr(Config().trainer, "model_name")
            else "custom"
        )
        filename = f"checkpoint_{model_name}_{self.current_round - 2}.pth"
        checkpoint_path = Config.params["checkpoint_path"]
        model_path = f"{checkpoint_path}/{filename}"

        initial_model = copy.deepcopy(self.trainer.model)
        initial_model.load_state_dict(torch.load(model_path))

        initial = torch.zeros(0)
        for __, weight in initial_model.cpu().state_dict().items():
            initial = torch.cat((initial, weight.view(-1)))

        current = torch.zeros(0)
        for __, weight in self.trainer.model.cpu().state_dict().items():
            current = torch.cat((current, weight.view(-1)))

        weights_received = [update.payload for update in self.updates]
        baseline_weights = self.algorithm.extract_weights()
        deltas_received = self.algorithm.compute_weight_deltas(
            baseline_weights, weights_received, cluster_id=None
        )

        for i, update in enumerate(deltas_received):
            client_id = updates[i].client_id
            if self.clients_similarity[client_id] is not None:
                continue
            deltas = torch.zeros(0)
            for __, delta in update.items():
                deltas = torch.cat((deltas, delta.view(-1)))

            similarity = (F.cosine_similarity(current - initial, deltas, dim=0) + 1) / 2
            self.clients_similarity[client_id] = similarity.item()

    def _convert_to_solver(self, client_training_times):
        """Transform useful dictionaries to solvable matrix."""
        observed_training_times = [
            training_time
            for training_time in client_training_times.values()
            if training_time not in (None, 0)
        ]
        observed_similarities = [
            similarity
            for similarity in self.clients_similarity.values()
            if similarity is not None
        ]

        default_training_time = (
            sum(observed_training_times) / len(observed_training_times)
            if observed_training_times
            else 0.0
        )
        default_similarity = (
            sum(observed_similarities) / len(observed_similarities)
            if observed_similarities
            else 0.5
        )

        missing_training_times = sum(
            1
            for training_time in client_training_times.values()
            if training_time in (None, 0)
        )
        missing_similarities = sum(
            1 for similarity in self.clients_similarity.values() if similarity is None
        )

        if missing_training_times or missing_similarities:
            logging.warning(
                "[%s] Optimization statistics are incomplete. "
                "Imputing %d missing training times and %d missing similarities.",
                self,
                missing_training_times,
                missing_similarities,
            )

        training_time_list = []
        similarity_list = []

        for client_id in self.clusters:
            training_time = client_training_times.get(client_id, 0)
            if training_time in (None, 0):
                training_time = default_training_time
            training_time_list.append(training_time)

            similarity = self.clients_similarity.get(client_id)
            if similarity is None:
                similarity = default_similarity
            similarity_list.append(similarity)

        # Anchor intervals between clusters: (max value - min value) / the number of clusters
        similarity_interval = (
            max(similarity_list) - min(similarity_list)
        ) / self.num_clusters

        training_time_interval = (
            max(training_time_list) - min(training_time_list)
        ) / self.num_clusters

        # Produce matrices containing the distances between input data and cluster anchors
        training_anchor_distances = []
        similarity_anchor_distances = []

        for cluster_id in range(self.num_clusters):
            training_anchor_distances.append(
                [
                    abs(
                        training_time_interval * cluster_id
                        + min(training_time_list)
                        - time
                    )
                    for time in training_time_list
                ]
            )
            similarity_anchor_distances.append(
                [
                    abs(
                        similarity_interval * cluster_id
                        + min(similarity_list)
                        - similarity
                    )
                    for similarity in similarity_list
                ]
            )

        training_time_np_matrix = numpy.matrix(training_anchor_distances)
        similarity_np_matrix = numpy.matrix(similarity_anchor_distances)

        # Use scaler to tune the weight of training time and similarity on opt problem.
        # The range of original training time is 0 - 30, similarity is 0 - 1.
        # If training_time_scaler = 1, and similarity_scaler = 10,
        # the weights of training time and similarity are both 1/2.
        training_time_scaler = 1
        similarity_scaler = 1

        training_time_matrix = training_time_np_matrix * training_time_scaler
        similarity_matrix = similarity_np_matrix * similarity_scaler

        # Transfer numpy.matrix to cvxopt.matrix for indexing.
        training_time_matrix = matrix(training_time_matrix)
        similarity_matrix = matrix(similarity_matrix)

        l2_list = []

        for client_id in self.clusters:
            for cluster_id in range(self.num_clusters):
                # The key of self.clusters are client_ids, the index start from 1
                # Thus, we use (client_id - 1) as the matrix index here.
                l2_array = numpy.array(
                    [
                        training_time_matrix[cluster_id, client_id - 1],
                        similarity_matrix[cluster_id, client_id - 1],
                    ]
                )

                # Calculate the norm.
                l2_norm = norm(l2_array, 2)
                l2_list.append(l2_norm)

        l2_matrix = matrix(l2_list, training_time_np_matrix.shape)

        return l2_matrix


    def _optimize_clustering(self, updates):
        """
        Computing optimized clustering with an optimization solver,
        based on both per-client training times and the cosine similarity,
        computed between the clients and the global anchors.
        """
        cluster_map_path = getattr(
            Config().server, "controlled_cluster_map_path", None
        )
        resolved_cluster_map_path = None
        if cluster_map_path:
            resolved_cluster_map_path = os.path.join(
                Config.params["base_path"], cluster_map_path
            )
            if os.path.exists(resolved_cluster_map_path):
                with open(
                    resolved_cluster_map_path, "r", encoding="utf-8"
                ) as cluster_file:
                    self.clusters = {
                        int(client_id): int(cluster_id)
                        for client_id, cluster_id in json.load(cluster_file).items()
                    }
                self.algorithm.init_clusters(self.clusters)
                logging.info(
                    "[%s] Reused controlled cluster map from %s: %s",
                    self,
                    resolved_cluster_map_path,
                    self.clusters,
                )
                return

        logging.info(
            fonts.colourize(
                f"\n[{self}] Starting the solve the optimization problem to assign "
                f"{len(self.clusters)} clients to {self.num_clusters} clusters.",
                colour="green",
            )
        )

        client_training_times = self._extract_training_times(updates)
        self._cosine_similarity(updates)

        assignment_list = solver.solve(
            # workload_max
            # (len(self.clusters) / self.num_clusters) * 2,
            # len(self.clusters) / self.num_clusters,
            len(self.clusters) / 2,
            # workload_min
            # len(self.clusters) / (self.num_clusters * 2),
            # len(self.clusters) / self.num_clusters,
            len(self.clusters) / self.num_clusters,
            1,
            self.num_clusters,
            len(self.clusters),
            self._convert_to_solver(client_training_times),
        )

        self._convert_from_solver(assignment_list)
        if resolved_cluster_map_path:
            os.makedirs(os.path.dirname(resolved_cluster_map_path), exist_ok=True)
            if not os.path.exists(resolved_cluster_map_path):
                temporary_path = (
                    resolved_cluster_map_path + f".{os.getpid()}.tmp"
                )
                serializable_clusters = {
                    int(client_id): int(cluster_id)
                    for client_id, cluster_id in self.clusters.items()
                }
                with open(temporary_path, "w", encoding="utf-8") as cluster_file:
                    json.dump(serializable_clusters, cluster_file, sort_keys=True)
                os.replace(temporary_path, resolved_cluster_map_path)
                logging.info(
                    "[%s] Saved controlled cluster map to %s.",
                    self,
                    resolved_cluster_map_path,
                )

    def _ensure_complete_cluster_assignments(self):
        """Ensure every client has a concrete cluster assignment."""
        missing_clients = [
            client_id
            for client_id in range(1, self.total_clients + 1)
            if self.clusters.get(client_id) is None
        ]

        if missing_clients:
            logging.warning(
                "[%s] Solver left %d clients without clusters. Applying fallback assignment.",
                self,
                len(missing_clients),
            )

            for client_id in missing_clients:
                self.clusters[client_id] = client_id % self.num_clusters

    def _convert_from_solver(self, assignment_list):
        """Convert the assignment that generate from solver to clients_clusters dictionary."""
        assignment_array = numpy.array(assignment_list)

        index_of_value_1 = numpy.array(numpy.argwhere(assignment_array == 1))

        self.clusters = dict(zip(index_of_value_1[:, 1] + 1, index_of_value_1[:, 0]))
        self._ensure_complete_cluster_assignments()
        self.algorithm.init_clusters(self.clusters)

        logging.info(
            fonts.colourize(
                f"\n[{self}] Cluster assignments: {self.clusters}", colour="green"
            )
        )
