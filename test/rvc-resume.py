"""Small stdlib regression fixtures; never read or write a real RVC installation."""
from pathlib import Path
from types import SimpleNamespace
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'py'))
import rvc_resume as r


class ResumeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='dsh-rvc-resume-')
        self.root = Path(self.tmp.name).resolve()
        self.repo = self.root/'engine'
        self.log = self.repo/'logs'/'fixture'
        self.log.mkdir(parents=True)
        self.work = self.root/'work'/'fixture'
        for name, value in [('G_100.pth', b'generator'), ('D_100.pth', b'discriminator'), ('config.json', b'{}'), ('filelist.txt', b'fixture')]:
            (self.log/name).write_bytes(value)

    def tearDown(self):
        self.assertEqual(self.root.parent, Path(tempfile.gettempdir()).resolve())
        self.assertTrue(self.root.name.startswith('dsh-rvc-resume-'))
        self.tmp.cleanup()

    def backup(self, **kwargs):
        return r.backup_resume(self.log, self.work, self.repo, 'fixture-job', 11,
                               inspect=kwargs.pop('inspect', lambda _: {'epoch': 10}), **kwargs)

    def test_independent_verified_pair_survives_cleanup(self):
        directory, manifest = self.backup()
        self.assertFalse(directory.is_relative_to(self.repo/'logs'))
        self.assertEqual(manifest['fromEpoch'], 10)
        self.assertTrue(manifest['complete'])
        for row in manifest['files']:
            self.assertEqual(r.sha(directory/row['file']), row['sha256'])
        (self.log/'G_100.pth').unlink()
        (self.log/'D_100.pth').unlink()
        self.assertEqual((directory/'G_100.pth').read_bytes(), b'generator')
        with self.assertRaises(FileExistsError):
            # Same snapshot ID is never silently overwritten.
            with patch.object(r.time, 'strftime', return_value=directory.name.rsplit('-fixture-job', 1)[0]):
                # Restore only our tiny fixture source files for the collision test.
                (self.log/'G_100.pth').write_bytes(b'generator')
                (self.log/'D_100.pth').write_bytes(b'discriminator')
                self.backup()

    def test_missing_pair_never_becomes_fresh_training(self):
        (self.log/'D_100.pth').unlink()
        with self.assertRaisesRegex(ValueError, '续训缺少 D_'):
            self.backup()
        self.assertFalse((self.work/'checkpoint-backups').exists())

    def test_mismatched_pair_and_epoch_fail(self):
        (self.log/'G_200.pth').write_bytes(b'newer')
        with self.assertRaisesRegex(ValueError, '编号不配对'):
            self.backup()
        (self.log/'G_200.pth').unlink()
        with self.assertRaisesRegex(ValueError, '实际训练轮次不一致'):
            self.backup(inspect=lambda p: {'epoch': 10 if p.name.startswith('G') else 9})
        manifest = next((self.work/'checkpoint-backups').glob('*/backup.json'))
        self.assertFalse(json.loads(manifest.read_text(encoding='utf-8'))['complete'])

    def test_low_space_or_copy_failure_cannot_complete_backup(self):
        with patch.object(r.shutil, 'disk_usage', return_value=SimpleNamespace(free=0)):
            with self.assertRaisesRegex(ValueError, '空间不足'):
                self.backup()
        with patch.object(r.os, 'fsync', side_effect=OSError('fixture disk full')):
            with self.assertRaisesRegex(OSError, 'disk full'):
                self.backup()
        manifest = next((self.work/'checkpoint-backups').glob('*/backup.json'))
        self.assertFalse(json.loads(manifest.read_text(encoding='utf-8'))['complete'])

    def test_backup_inside_engine_logs_is_rejected(self):
        self.work = self.log/'work'
        with self.assertRaisesRegex(ValueError, 'logs 目录之外'):
            self.backup()

    def test_hash_mismatch_leaves_incomplete_snapshot(self):
        with patch.object(r, 'sha', return_value='0'*64):
            with self.assertRaisesRegex(ValueError, '校验失败'):
                self.backup()
        manifest = next((self.work/'checkpoint-backups').glob('*/backup.json'))
        self.assertFalse(json.loads(manifest.read_text(encoding='utf-8'))['complete'])

    def test_checkpoint_requires_optimizer_and_real_epoch(self):
        value = {'model': {'weight': 1}, 'iteration': 10, 'learning_rate': .0001,
                 'optimizer': {'param_groups': [{}], 'state': {0: 1}}}
        self.assertEqual(r.checkpoint_metadata('fixture', lambda _: value)['epoch'], 10)
        for key, invalid in [('optimizer', {}), ('iteration', 0), ('learning_rate', float('nan'))]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                r.checkpoint_metadata('fixture', lambda _: {**value, key: invalid})

    def guard_fixture(self, invalid=False):
        backup, manifest = self.backup()
        cfg = self.root/'guard.json'
        proof = self.root/'proof.json'
        r.save(cfg, {**manifest, 'backupDir': str(backup), 'proofPath': str(proof)})
        tensor = SimpleNamespace(shape=(1,))
        model = SimpleNamespace(state_dict=lambda: {'w': tensor}, load_state_dict=lambda value, strict: None)
        optimizer = SimpleNamespace(load_state_dict=lambda _: None)
        load = lambda _: {'model': {'w': SimpleNamespace(shape=(2,) if invalid else (1,))},
                          'iteration': 10, 'optimizer': {'state': {0: 1}}, 'learning_rate': .0001}
        utils = SimpleNamespace()
        r.install_guard(cfg, utils, load)
        return backup, proof, utils, model, optimizer

    def test_restore_uses_backup_and_confirms_both_optimizers(self):
        backup, proof, utils, model, opt = self.guard_fixture()
        (self.log/'G_100.pth').unlink()
        (self.log/'D_100.pth').unlink()
        for kind in ['D', 'G']:
            selected = utils.latest_checkpoint_path(str(self.log), kind+'_*.pth')
            self.assertEqual(Path(selected).parent, backup)
            result = utils.load_checkpoint(selected, model, opt)
            self.assertEqual(result[3], 10)
        p = json.loads(proof.read_text(encoding='utf-8'))
        self.assertTrue(p['confirmed'])
        self.assertTrue(all(x['optimizerRestored'] for x in p['loaded'].values()))

    def test_restore_failure_bypasses_upstream_restart_fallback(self):
        backup, proof, utils, model, opt = self.guard_fixture(invalid=True)
        restarted = False
        with self.assertRaisesRegex(r.ResumeAbort, '禁止回退第 1 轮'):
            try:
                utils.load_checkpoint(backup/'D_100.pth', model, opt)
            except Exception:
                restarted = True
        self.assertFalse(restarted)
        self.assertFalse(proof.exists())

    def test_tampered_snapshot_is_not_loaded(self):
        backup, proof, utils, model, opt = self.guard_fixture()
        (backup/'G_100.pth').write_bytes(b'changed after verification')
        with self.assertRaisesRegex(r.ResumeAbort, 'SHA-256'):
            utils.load_checkpoint(backup/'G_100.pth', model, opt)
        self.assertFalse(proof.exists())


if __name__ == '__main__':
    unittest.main()
