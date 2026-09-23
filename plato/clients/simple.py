# MGA local research fork; includes modifications relative to upstream Plato/KNOT.
# Distributed under Apache-2.0; see LICENSE, NOTICE and MODIFICATIONS.md.
"""
A basic federated learning client who sends weight updates to the server.
"""

import logging
import os
import random
import time
from types import SimpleNamespace

from plato.algorithms import registry as algorithms_registry
from plato.clients import base
from plato.config import Config
from plato.datasources import registry as datasources_registry
from plato.processors import registry as processor_registry
from plato.samplers import registry as samplers_registry
from plato.trainers import registry as trainers_registry
from plato.utils import fonts
import numpy as np
import torch

class Client(base.Client):
    """A basic federated learning client who sends simple weight updates."""

    def __init__(
        self,
        model=None,
        datasource=None,
        algorithm=None,
        trainer=None,
        callbacks=None,
        trainer_callbacks=None,
    ):
        super().__init__(callbacks=callbacks)
        # Save the callbacks that will be passed to trainer later
        self.trainer_callbacks = trainer_callbacks

        self.custom_model = model
        self.model = None

        self.custom_datasource = datasource
        self.datasource = None

        self.custom_algorithm = algorithm
        self.algorithm = None

        self.custom_trainer = trainer
        self.trainer = None

        self.trainset = None  # Training dataset
        self.testset = None  # Testing dataset
        self.sampler = None
        self.testset_sampler = None  # Sampler for the test set

        self._report = None
        self.data_deletion_round = Config().clients.data_deletion_round
        self.clients_requesting_deletion = Config().clients.clients_requesting_deletion
        self.total_clients = Config().clients.total_clients
        
        self._ensure_pth_rec()
        

    def _ensure_pth_rec(self):
        """Create pth_rec.npy once, without clobbering other client records."""
        from plato.utils.pth_rec import ensure_pth_rec

        ensure_pth_rec("pth_rec.npy", self.total_clients)

    def configure(self) -> None:
        """Prepares this client for training."""
        super().configure()

        if self.model is None and self.custom_model is not None:
            self.model = self.custom_model

        if self.trainer is None and self.custom_trainer is None:
            self.trainer = trainers_registry.get(
                model=self.model, callbacks=self.trainer_callbacks
            )
        elif self.trainer is None and self.custom_trainer is not None:
            self.trainer = self.custom_trainer(
                model=self.model, callbacks=self.trainer_callbacks
            )

        self.trainer.set_client_id(self.client_id)

        if self.algorithm is None and self.custom_algorithm is None:
            self.algorithm = algorithms_registry.get(trainer=self.trainer)
        elif self.algorithm is None and self.custom_algorithm is not None:
            self.algorithm = self.custom_algorithm(trainer=self.trainer)

        self.algorithm.set_client_id(self.client_id)

        # Pass inbound and outbound data payloads through processors for
        # additional data processing
        self.outbound_processor, self.inbound_processor = processor_registry.get(
            "Client", client_id=self.client_id, trainer=self.trainer
        )

        # Setting up the data sampler
        if self.datasource:
            self.sampler = samplers_registry.get(self.datasource, self.client_id)

            if (
                hasattr(Config().clients, "do_test")
                and Config().clients.do_test
                and hasattr(Config().data, "testset_sampler")
            ):
                # Set the sampler for test set
                self.testset_sampler = samplers_registry.get(
                    self.datasource, self.client_id, testing=True
                )

    def _load_data(self) -> None:
        """Generates data and loads them onto this client."""
        # The only case where Config().data.reload_data is set to true is
        # when clients with different client IDs need to load from different datasets,
        # such as in the pre-partitioned Federated EMNIST dataset. We do not support
        # reloading data from a custom datasource at this time.
        if (
            self.datasource is None
            or hasattr(Config().data, "reload_data")
            and Config().data.reload_data
        ):
            logging.info("[%s] Loading its data source...", self)

            if self.custom_datasource is None:
                self.datasource = datasources_registry.get(client_id=self.client_id)
            elif self.custom_datasource is not None:
                self.datasource = self.custom_datasource()

            logging.info(
                "[%s] Dataset size: %s", self, self.datasource.num_train_examples()
            )

    def _allocate_data(self) -> None:
        """Allocate training or testing dataset of this client."""
        if hasattr(Config().trainer, "use_mindspore"):
            # MindSpore requires samplers to be used while constructing
            # the dataset
            self.trainset = self.datasource.get_train_set(self.sampler)
        else:
            # PyTorch uses samplers when loading data with a data loader
            self.trainset = self.datasource.get_train_set()

        if hasattr(Config().clients, "do_test") and Config().clients.do_test:
            # Set the testset if local testing is needed
            self.testset = self.datasource.get_test_set()

    def _load_payload(self, server_payload) -> None:
        """Loads the server model onto this client."""
        self.algorithm.load_weights(server_payload)

    async def _train(self):
        """The machine learning training workload on a client."""
        logging.info(
            fonts.colourize(
                f"[{self}] Started training in communication round #{self.current_round}."
            )
        )

        # Perform model training
        deterministic_seed = (
            int(getattr(Config().trainer, "random_seed", 1))
            + int(self.current_round) * 100000
            + int(self.client_id)
        )
        random.seed(deterministic_seed)
        np.random.seed(deterministic_seed % (2**32 - 1))
        torch.manual_seed(deterministic_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(deterministic_seed)

        try:
            if hasattr(self.trainer, "current_round"):
                self.trainer.current_round = self.current_round

            # judge if the client need to unlearn

            retrain = bool(
                getattr(Config().args, "scratch", False)
                or getattr(Config().trainer, "knot_retrain", False)
            )
            if retrain:
                training_time = self.trainer.train(self.trainset, self.sampler)
            else:
                unlearning_window = getattr(Config().trainer, "unlearning_window", None)
                if unlearning_window is None:
                    in_unlearning_window = self.current_round >= self.data_deletion_round
                else:
                    in_unlearning_window = (
                        self.data_deletion_round
                        <= self.current_round
                        < self.data_deletion_round + int(unlearning_window)
                    )
                unlearning = (
                    in_unlearning_window
                    and self.client_id in self.clients_requesting_deletion
                )
                phase = getattr(self, "mga_phase", None)
                if phase is not None:
                    if phase not in ("train", "unlearn", "recover"):
                        raise ValueError("Invalid explicit MGA dispatch phase.")
                    # A large target set may take multiple dispatch batches;
                    # the phase, not a legacy round window, owns this decision.
                    unlearning = phase == "unlearn"
                local_epochs = getattr(self, "local_epochs", None)
                training_time = self.trainer.train(
                    self.trainset,
                    self.sampler,
                    unlearning,
                    unlearning_epochs=getattr(self, "unlearning_epochs", None),
                    local_epochs=local_epochs,
                )

        except ValueError as exc:
            logging.info(
                fonts.colourize(f"[{self}] Error occurred during training: {exc}")
            )
            await self.sio.disconnect()
            raise  # Never return old weights as a successful training response.

        # Extract model weights and biases
        weights = self.algorithm.extract_weights()

        # Generate a report for the server, performing model testing if applicable
        if (hasattr(Config().clients, "do_test") and Config().clients.do_test) and (
            not hasattr(Config().clients, "test_interval")
            or self.current_round % Config().clients.test_interval == 0
        ):
            accuracy = self.trainer.test(self.testset, self.testset_sampler)

            if accuracy == -1:
                # The testing process failed, disconnect from the server
                logging.info(
                    fonts.colourize(
                        f"[{self}] Accuracy is -1 when testing. Disconnecting from the server."
                    )
                )
                await self.sio.disconnect()

            if hasattr(Config().trainer, "target_perplexity"):
                logging.info("[%s] Test perplexity: %.2f", self, accuracy)
            else:
                logging.info("[%s] Test accuracy: %.2f%%", self, 100 * accuracy)
        else:
            accuracy = 0

        comm_time = time.time()

        if (
            hasattr(Config().clients, "sleep_simulation")
            and Config().clients.sleep_simulation
        ):
            sleep_seconds = Config().client_sleep_times[self.client_id - 1]
            avg_training_time = Config().clients.avg_training_time

            training_time = (
                avg_training_time + sleep_seconds
            ) * Config().trainer.epochs

        report = SimpleNamespace(
            client_id=self.client_id,
            num_samples=self.sampler.num_samples(),
            accuracy=accuracy,
            training_time=training_time,
            comm_time=comm_time,
            update_response=False,
        )

        self._report = self.customize_report(report)

        return self._report, weights

    async def _obtain_model_update(self, client_id, requested_time):
        """Retrieves a model update corresponding to a particular wall clock time."""
        model = self.trainer.obtain_model_update(client_id, requested_time)
        weights = self.algorithm.extract_weights(model)
        self._report.comm_time = time.time()
        self._report.client_id = client_id
        self._report.update_response = True

        return self._report, weights

    def customize_report(self, report: SimpleNamespace) -> SimpleNamespace:
        """Customizes the report with any additional information."""
        return report
