"""Exercise the single training phase with a small compatible SNGP model."""
import copy
from pathlib import Path
import random
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

CONTINUAL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CONTINUAL))

from frozen_support.models import RandomFourierFeatures, SNGPHead
import launch_suite
from replay_memory import ReplayMemory
import training_engine as engine
import worker


class TinySNGP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Linear(3, 3)
        self.rff = RandomFourierFeatures(3, 4, gamma=.25)
        self.gp_head = SNGPHead(4, 3)

    def features(self, inputs):
        return self.encoder(inputs).tanh()

    def forward(self, inputs):
        native = self.rff(self.features(inputs))
        logits, std = self.gp_head(native)
        return native, logits, std


class SinglePhaseTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)

    def test_stage_uses_one_optimizer_and_rebuilds_training_precision(self):
        model = TinySNGP()
        source = torch.tensor([[1., .5, -.5], [-.5, 1., .25]])
        target = torch.tensor([[.2, -.5, 1.], [1., -.2, .5]])
        train = {
            'source': DataLoader(TensorDataset(source, torch.tensor([0, 1])), batch_size=2),
            'target': DataLoader(TensorDataset(target, torch.tensor([-1, 2])), batch_size=2),
        }
        validation = {
            'source': DataLoader(TensorDataset(torch.tensor([[.5, .2, -.1]]),
                torch.tensor([0]), torch.tensor([1])), batch_size=1),
            'target': DataLoader(TensorDataset(torch.tensor([[-.2, .5, 1.]]),
                torch.tensor([2]), torch.tensor([1])), batch_size=1),
        }
        train_ids = {'source': ['train_s0', 'train_s1'], 'target': ['train_t0', 'train_t1']}
        validation_ids = {'source': ['validation_s'], 'target': ['validation_t']}
        before = model.encoder.weight.detach().clone()
        events = []
        adam = torch.optim.Adam
        config = dict(max_epochs=1, warmup_epochs=1, patience=2,
            initial_candidate=False, learning_rate=.01, lambda_kd=0.,
            inner_kwargs=dict(lambda_mmd=0., lambda_classwise_mmd=0.,
                lambda_pseudo=0., use_classwise_mmd=False, use_pseudo_labels=False))
        with patch.object(torch.optim, 'Adam', wraps=adam) as optimizer, \
                patch.object(torch.optim, 'AdamW', side_effect=AssertionError('Unexpected second optimizer')):
            result = engine.train_stage(model, train, validation, 'cpu',
                train_ids=train_ids, validation_ids=validation_ids,
                config=config, callback=events.append)
        self.assertEqual(optimizer.call_count, 1)
        self.assertEqual(result['epochs_run'], 1)
        self.assertEqual(result['selected_epoch'], 1)
        self.assertEqual(result['selection_phase'], 'base')
        self.assertEqual([row['epoch'] for row in events], [0, 1])
        self.assertTrue(all(row['phase'] == 'base' for row in result['history']))
        self.assertEqual(sum(row.get('optimizer_steps', 0) for row in result['history']), 1)
        self.assertFalse(any(key.startswith('geometry_') for key in result))
        self.assertTrue(model.encoder.weight.requires_grad)
        self.assertFalse(torch.equal(before, model.encoder.weight))
        self.assertEqual(result['precision_seen'], 4)
        with torch.no_grad():
            native = model.rff(model.features(torch.cat([source, target])))
            expected = torch.eye(4) * model.gp_head.ridge_penalty + native.T @ native
        self.assertTrue(torch.allclose(model.gp_head.precision_matrix, expected, atol=1e-6, rtol=1e-6))

    def test_method_configs_keep_retention_terms_in_the_training_phase(self):
        recipe = launch_suite.training_recipe()
        for method in launch_suite.METHODS:
            for stage in range(1, 5):
                with self.subTest(method=method, stage=stage):
                    cfg = engine.effective_config(worker.base_phase_config({'training': recipe}, method, stage))
                    self.assertFalse(any(key.startswith('geometry_') for key in cfg))
                    self.assertEqual(cfg['temperature'], 2.)
                    self.assertEqual(cfg['lambda_kd'], 3. if stage > 1 and method in {'continual', 'lwf'} else 0.)
                    self.assertEqual(cfg['lambda_feature_kd'], 1. if stage > 1 and method == 'continual' else 0.)
                    self.assertEqual(cfg['kd_current'], stage > 1 and method == 'lwf')
                    self.assertEqual(cfg['learning_rate'], 5e-4 if stage == 1 or method == 'joint_retraining' else 2e-5)

    def test_removed_training_options_are_rejected(self):
        for key, value in [('geometry_enabled', True), ('geometry_max_epochs', 1),
                           ('lambda_compactness', 1.), ('lambda_supcon', .1)]:
            with self.subTest(setting=key):
                with self.assertRaisesRegex(ValueError, 'Unknown stage settings'):
                    engine.effective_config({key: value})

    def test_replay_feature_distillation_keeps_encoder_gradients(self):
        model = TinySNGP().eval()
        teacher = copy.deepcopy(model).eval()
        with torch.no_grad():
            teacher.encoder.weight.add_(.2)
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)
        memories = {}
        inputs = {'source': torch.tensor([[1., .5, -.5]]),
                  'target': torch.tensor([[-.5, 1., .25]])}
        for domain, labels in [('source', [0]), ('target', [-1])]:
            scope = SimpleNamespace(domain=domain, split='train', block=1,
                key='old_' + domain, ids=['old_' + domain],
                training_y=torch.tensor(labels), x=inputs[domain])
            memory = ReplayMemory(domain, capacity=1)
            memory.add([(scope, 0)])
            memories[domain] = memory
        cfg = engine.effective_config(dict(lambda_kd=3., lambda_feature_kd=1., feature_scale=2.))
        actual = engine.replay_objective(model, teacher, memories, torch.device('cpu'), random.Random(42), cfg)
        expected = sum(F.mse_loss(model.features(inputs[domain]), teacher.features(inputs[domain]))
                       for domain in ('source', 'target')) / 2 / 4
        self.assertTrue(torch.allclose(actual['feature_kd'], expected, atol=1e-7, rtol=1e-6))
        self.assertGreater(float(actual['feature_kd'].detach()), 0.)
        self.assertGreater(float(actual['kd'].detach()), 0.)
        actual['feature_kd'].backward()
        self.assertTrue(bool(torch.isfinite(model.encoder.weight.grad).all()))
        self.assertGreater(float(model.encoder.weight.grad.abs().sum()), 0.)
        self.assertTrue(all(parameter.grad is None for parameter in teacher.parameters()))


if __name__ == '__main__':
    unittest.main()
