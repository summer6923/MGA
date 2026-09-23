# MGA local research fork; includes modifications relative to upstream Plato/KNOT.
# Distributed under Apache-2.0; see LICENSE, NOTICE and MODIFICATIONS.md.
"""Regression tests for the real dispatch hooks and client-local poison operator.
Run: python tests/test_mga_pipeline.py (no dataset download or GPU required).
"""
from pathlib import Path
import asyncio
import copy
import json
import random
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch, Mock, AsyncMock

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CONFIG_TMP = tempfile.TemporaryDirectory(prefix="mga_unit_")
CONFIG_PATH = Path(CONFIG_TMP.name) / "test.yml"
CONFIG_PATH.write_text(yaml.safe_dump({
    "general": {"base_path": CONFIG_TMP.name},
    "clients": {"type": "simple", "total_clients": 8, "per_round": 3,
                "data_deletion_round": 5, "clients_requesting_deletion": [1, 2, 3, 4],
                "deleted_data_ratio": 0.5, "backdoor_enabled": True,
                "backdoor_poison_rate": 0.5, "backdoor_target_label": 1,
                "random_seed": 17},
    "server": {"address": "127.0.0.1", "port": 39301, "clusters": 2,
               "synchronous": False, "simulate_wall_time": True,
               "exclude_delete_clients_after_deletion": True,
               "skip_cluster_rollback_after_deletion": True,
               "retained_fedavg_after_deletion": True,
               "retained_recovery_epochs_after_deletion": 2,
               "do_test": False, "do_optimized_clustering": False},
    "trainer": {"type": "basic", "model_name": "lenet5", "epochs": 1,
                "rounds": 8, "batch_size": 4, "optimizer": "SGD",
                "mga_enabled": True, "unlearning_strategy": "mga",
                "mga_forget_scope": "client", "mga_radius": 0.5,
                "dynamic_e_min": 1, "dynamic_e_max": 5, "max_concurrency": 1},
    "data": {"datasource": "MNIST", "sampler": "iid", "partition_size": 8,
             "random_seed": 17},
    "parameters": {"optimizer": {"lr": 0.02, "momentum": 0.9}},
    "algorithm": {"type": "fedavg"},
    "results": {"result_path": "results"},
}), encoding="utf8")
sys.argv = [str(Path(__file__)), "-c", str(CONFIG_PATH)]
from plato.config import Config
from mga.data import PoisonSpec, PoisonedDataset, IndexSampler, select_subset, invert_trigger
from mga.reverse import reverse_optimize
from mga.protocol import PendingTargets, combine_states, clone_state, enabled_for_config
from mga.server import Server
from mga.client import Client
from plato.clients import simple
from plato.trainers.basic import Trainer


class Scalar(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(1))
    def forward(self, x):
        return self.w.expand_as(x)


class DataTests(unittest.TestCase):
    def setUp(self):
        raw = torch.arange(80).float().reshape(10, 1, 2, 4) / 80
        self.raw = raw
        self.dataset = TensorDataset((raw - .1307) / .3081, torch.arange(10))
        self.spec = PoisonSpec(True, .5, 1, 17, "MNIST")
    def test_exact_rate_deterministic_membership(self):
        a = PoisonedDataset(self.dataset, range(10), self.spec, 1, True)
        b = PoisonedDataset(self.dataset, reversed(range(10)), self.spec, 1, True)
        self.assertEqual(a.poison_indices, b.poison_indices)
        self.assertEqual(len(a.poison_indices), 5)
    def test_input_and_label_changed_without_mutating_dataset(self):
        view = PoisonedDataset(self.dataset, range(10), self.spec, 1, True)
        i = next(iter(view.poison_indices)); original = self.dataset[i][0].clone()
        x, y = view[i]
        torch.testing.assert_close(x, ((1 - self.raw[i]) - .1307) / .3081)
        self.assertEqual(int(y), 1)
        self.assertTrue(torch.equal(self.dataset[i][0], original))
        self.assertEqual(int(self.dataset[i][1]), i)
        self.assertEqual(view.poison_read_count, 1)
    def test_retained_and_disabled_are_clean(self):
        for active, enabled in [(False, True), (True, False)]:
            spec = PoisonSpec(enabled, 1., 1, 17, "MNIST")
            view = PoisonedDataset(self.dataset, range(10), spec, 1, active)
            self.assertFalse(view.poison_indices)
            for i in range(10):
                x, y = view[i]
                self.assertTrue(torch.equal(x, self.dataset[i][0]))
                self.assertEqual(int(y), i)
    def test_fraction_extremes_and_validation(self):
        self.assertEqual(len(select_subset(range(10), 0, 1, 1)), 0)
        self.assertEqual(len(select_subset(range(10), 1, 1, 1)), 10)
        for rate in [-.1, 1.1, float("nan")]:
            with self.assertRaises(ValueError): select_subset(range(10), rate, 1, 1)
        with self.assertRaises(ValueError): PoisonSpec(True, .5, 10, 1, "MNIST")
        with self.assertRaises(ValueError): PoisonSpec(True, .5, 1, 1, "unknown")
    def test_deletion_subset_preserves_original_partition(self):
        original = [3, 4, 8, 9]
        removed = select_subset(original, .5, 17, 1)
        kept = set(original) - removed
        self.assertEqual(removed | kept, set(original))
        self.assertFalse(removed & kept)
        self.assertEqual(set(IndexSampler(removed, 1).get()), removed)
        with self.assertRaises(ValueError): IndexSampler([], 1)
    def test_escaped_partition_and_double_wrapper_rejected(self):
        view = PoisonedDataset(self.dataset, [1, 2], self.spec, 1, True)
        with self.assertRaises(IndexError): view[3]
        with self.assertRaises(ValueError): PoisonedDataset(view, [1, 2], self.spec, 1, True)
    def test_cifar_and_purchase_inversion(self):
        x = torch.rand(2, 3, 4, 4)
        mu = torch.tensor([.485,.456,.406]).view(3,1,1)
        sd = torch.tensor([.229,.224,.225]).view(3,1,1)
        torch.testing.assert_close(invert_trigger((x-mu)/sd, "CIFAR-10"), (1-x-mu)/sd)
        features = torch.tensor([0., 1., 0.])
        self.assertTrue(torch.equal(invert_trigger(features, "Purchase"), 1-features))
    def test_membership_does_not_change_global_rng(self):
        state = random.getstate(); tstate = torch.random.get_rng_state().clone()
        select_subset(range(100), .8, 17, 1)
        self.assertEqual(state, random.getstate())
        self.assertTrue(torch.equal(tstate, torch.random.get_rng_state()))


class ReverseTests(unittest.TestCase):
    def run_linear(self, E, lr=.2, radius=1.):
        model = Scalar(); reference = clone_state(model.state_dict())
        data = DataLoader(TensorDataset(torch.ones(5,1), torch.zeros(5)), batch_size=2)
        result = reverse_optimize(model, reference, data, lambda p,y:p.mean(), E,lr,radius,"cpu")
        return model, result
    def test_reject_infeasible_instead_of_project(self):
        model, result = self.run_linear(4)
        self.assertAlmostEqual(float(model.w.detach()), .8, places=6)
        self.assertEqual(result["accepted_steps"], 2)
        self.assertEqual(result["attempted_steps"], 3)
        self.assertEqual(result["stop_reason"], "outside_radius")
    def test_larger_budget_boundary_rule(self):
        model, result = self.run_linear(9)
        self.assertAlmostEqual(float(model.w.detach()), .6, places=6)
        self.assertEqual(result["accepted_steps"], 1)
    def test_loss_increases_and_full_gradient_count(self):
        model, result = self.run_linear(2, .01, 1.)
        self.assertGreater(result["loss_after"], result["loss_before"])
        self.assertEqual(result["gradient_examples"], 10)
        self.assertEqual(result["accepted_steps"], 2)
        self.assertTrue(model.training)
    def test_zero_gradient(self):
        m=Scalar();data=DataLoader(TensorDataset(torch.ones(2,1),torch.zeros(2)),batch_size=2)
        result=reverse_optimize(m,clone_state(m.state_dict()),data,lambda p,y:(p*0).mean(),2,.01,.5,"cpu")
        self.assertEqual(result["stop_reason"],"zero_gradient")
        self.assertEqual(result["returned_distance"],0)
    def test_invalid_numeric_and_empty_inputs(self):
        for lr,rad in [(0,1),(.1,0),(float("nan"),1)]:
            with self.assertRaises(ValueError): self.run_linear(2,lr,rad)
        m=Scalar()
        with self.assertRaises(ValueError):
            reverse_optimize(m,clone_state(m.state_dict()),[],lambda p,y:p.mean(),1,.1,1,"cpu")


class FakeAlgorithm:
    def __init__(self): self.states={}
    def load_weights(self,state,cluster_id=None): self.states[cluster_id]=clone_state(state)


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix="mga_protocol_test_")
        self.s=Server.__new__(Server)
        self.s._mga_enabled=True;self.s._mga_phase="unlearn";self.s._mga_reverse_round=True
        self.s._mga_dir=Path(self.tmp.name);self.s._mga_request=PendingTargets([1,2,3,4],"req")
        self.s._mga_before={"w":torch.tensor([0.])};self.s._mga_reference={"w":torch.tensor([1.])}
        self.s._mga_return_weights={i:.25 for i in [1,2,3,4]}
        self.s.algorithm=FakeAlgorithm();self.s.num_clusters=2;self.s.current_round=5
        self.s.prng_state=random.Random(1).getstate();self.s._true_retrain_reset_done=False
    def tearDown(self): self.tmp.cleanup()
    def update(self,i,req="req",phase="unlearn",evidence=True):
        report=NS(mga_phase=phase,mga_request_id=req,mga_dispatch_id=f"1_5_{i}")
        report.mga_execution={"phase":phase,"request_id":req,"dispatch_id":f"1_5_{i}",
             "client_id":i,"reverse":{"gradient_examples":4,"accepted_steps":1}} if evidence else None
        return NS(client_id=i,report=report,payload={"w":torch.tensor([2.])})
    def test_targets_selected_despite_exclusion_flag(self):
        self.assertEqual(self.s.choose_clients(list(range(1,9)),3),[1,2,3])
    def test_all_returns_required_before_recovery(self):
        before=asyncio.run(self.s.aggregate_weights([self.update(1),self.update(2),self.update(3)],None,None))
        self.assertEqual(float(before["w"]),0)
        self.assertEqual(self.s._mga_phase,"unlearn")
        self.assertEqual(self.s.choose_clients(list(range(1,9)),1),[4])
        result=asyncio.run(self.s.aggregate_weights([self.update(4)],None,None))
        self.assertEqual(float(result["w"]),2)
        self.assertEqual(self.s._mga_phase,"recover")
        selected=self.s.choose_clients(list(range(1,9)),8)
        self.assertEqual(set(selected),{5,6,7,8})
    def test_wrong_stale_retained_and_duplicate_returns_fail(self):
        for update in [self.update(1,"old"),self.update(5),self.update(1,phase="train"),self.update(1,evidence=False)]:
            with self.assertRaises((ValueError,RuntimeError)): self.s._mga_accept_returns([update])
        self.s._mga_accept_returns([self.update(1)])
        with self.assertRaises(ValueError): self.s._mga_accept_returns([self.update(1)])
    def test_reverse_output_is_not_overwritten_by_cluster_test(self):
        self.s.server_aggregation_count=7
        self.s.weights_aggregated([self.update(1)])
        self.assertEqual(self.s.server_aggregation_count,7)
        self.s.clients_processed()  # must not enter legacy rollback
    def test_small_last_batch_not_waiting_for_normal_threshold(self):
        s=self.s;s.selected_clients=[4];s._mga_batch_reports={};s.wall_time=0
        s._process_reports=AsyncMock();s.wrap_up=AsyncMock();s._select_clients=AsyncMock()
        x=(1.,4,{"client_id":4,"report":self.update(4).report,"payload":self.update(4).payload})
        asyncio.run(s._process_clients(x))
        s._process_reports.assert_awaited_once();s._select_clients.assert_awaited_once()
    def test_old_inflight_returns_are_drained_not_aggregated(self):
        s=self.s;s._mga_phase="drain";s._mga_drain_finish=0;s.training_clients={2:{}}
        s.reported_clients=["old"];s._select_clients=AsyncMock()
        x=(2.,1,{"client_id":1})
        asyncio.run(s._process_clients(x));s._select_clients.assert_not_awaited()
        self.assertEqual(s.reported_clients,[])
        s.training_clients={};asyncio.run(s._process_clients(x));s._select_clients.assert_awaited_once()
    def test_recovery_rejects_untagged_or_target_updates(self):
        self.s._mga_phase="recover";self.s._mga_reverse_round=False
        for update in [self.update(1,phase="recover"),self.update(5,phase="train")]:
            with self.assertRaises(RuntimeError): asyncio.run(self.s.aggregate_weights([update],None,None))
    def test_combination_rejects_nonfinite_and_preserves_integer_buffer(self):
        t={"w":torch.tensor([0.]),"n":torch.tensor(1)}
        r=combine_states(t,[{"w":torch.tensor([2.]),"n":torch.tensor(5)}],[1.])
        self.assertEqual(int(r["n"]),1)
        with self.assertRaises(ValueError):combine_states(t,[{"w":torch.tensor([float("nan")]),"n":torch.tensor(1)}],[1.])
    def test_retrain_switch_does_not_enable_mga(self):
        cfg=NS(args=NS(scratch=True),trainer=NS(mga_enabled=True,knot_retrain=False),server=NS())
        self.assertFalse(enabled_for_config(cfg))


class ClientAndProcessTests(unittest.TestCase):
    def test_startup_without_datasource_does_not_sample(self):
        client=Client.__new__(Client);client.trainer=NS();client.sampler=None
        with patch.object(simple.Client,"configure",return_value=None):client.configure()
        self.assertIsNone(client._mga_partition)
        self.assertIsNone(client.trainer.mga_context)
    def test_worker_reuse_clears_target_phase_and_epoch_state(self):
        c=Client.__new__(Client);c.client_id=1
        c.process_server_response({"mga_phase":"unlearn","unlearning_epochs":2,"local_epochs":9,"sampler_mode":"forget"})
        c.client_id=6;c.process_server_response({"mga_phase":"recover","local_epochs":2})
        self.assertIsNone(c._pending_knot_sampler_mode)
        self.assertEqual(c.local_epochs,2)
        c.process_server_response({"mga_phase":"train"})
        self.assertIsNone(c.local_epochs)
    def test_failed_child_cannot_reuse_existing_model(self):
        trainer=NS(train_process=Mock(),client_id=1,load_model=Mock())
        child=Mock(exitcode=1)
        with patch("plato.trainers.basic.mp.Process",return_value=child),patch("plato.trainers.basic.mp.get_start_method",return_value="spawn"):
            with self.assertRaises(ValueError):Trainer.train(trainer,None,None,False)
        trainer.load_model.assert_not_called()


if __name__ == "__main__":
    unittest.main(argv=[str(Path(__file__))], verbosity=2)
