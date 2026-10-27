from pathlib import Path
import hashlib
import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'py'))
import applio_route as a
import rvc_resume as r


class ApplioTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='dsh-applio-cli-')
        self.root=Path(self.tmp.name).resolve()
        self.repo=self.root/'engine with spaces'
        self.repo.mkdir()
        self.log=self.root/'moved logs'/'voice'
        self.log.mkdir(parents=True)
        self.c={'processes':2,'cutMode':'Automatic','normalization':'none','f0Method':'rmvpe','embedderDir':str(self.root/'local embedder'),
                'saveEvery':5,'epochs':30,'batchSize':2,'pretrainedG':str(self.root/'G.pth'),'pretrainedD':str(self.root/'D.pth'),'vocoder':'HiFi-GAN'}

    def tearDown(self):
        self.assertEqual(self.root.parent,Path(tempfile.gettempdir()).resolve())
        self.assertTrue(self.root.name.startswith('dsh-applio-cli-'))
        self.tmp.cleanup()

    def test_cli_training_disables_cleanup_and_shutdown_is_not_dispatched(self):
        cmd=a.command_for('run',self.c,'voice',self.repo,self.log,48000,{'epochs':40})
        self.assertEqual(cmd[14],'False')  # Native sys.argv[12] cleanup.
        self.assertEqual(cmd[5],'40')
        self.assertEqual(cmd[8],'0')
        self.assertEqual(len(cmd),17)
        self.assertEqual(Path(cmd[2]),self.repo/'rvc/train/train.py')

    def test_extract_pins_local_embedder_and_structured_paths(self):
        cmd=a.command_for('preprocess',self.c,'voice',self.repo,self.log,48000,{})
        self.assertEqual(cmd[3],str(self.log))
        self.assertEqual(cmd[8],'custom')
        self.assertEqual(cmd[9],self.c['embedderDir'])
        self.assertEqual(cmd[10],'2')
        with self.assertRaises(ValueError):
            a.command_for('infer',self.c,'voice',self.repo,self.log,48000,{})

    def test_external_running_experiment_is_not_adopted(self):
        (self.log/'config.json').write_text(json.dumps({'process_pids':[os.getpid()]}),encoding='utf-8')
        with self.assertRaisesRegex(ValueError,'外部训练进程'):
            a.inactive_external(self.log)

    def test_interface_changes_are_refused_and_eol_changes_are_accepted(self):
        source=self.repo/'entry.py';source.write_bytes(b'print(1)\r\n')
        contract=self.root/'contract.json'
        contract.write_text(json.dumps({'files':{'entry.py':hashlib.sha256(b'print(1)\n').hexdigest()}}),encoding='utf-8')
        with patch.object(a,'CONTRACT',contract):
            self.assertTrue(a.interfaces(self.repo)[1][0]['ok'])
            source.write_bytes(b'print(2)\n')
            with self.assertRaisesRegex(ValueError,'接口与已核验版本不一致'):
                a.require_interfaces(self.repo)

    def test_copied_applio_entry_keeps_native_module_search_path(self):
        import subprocess
        (self.repo/'rvc/train').mkdir(parents=True)
        (self.repo/'rvc/train/data_utils.py').write_text('MARKER="native_import_ok"\n',encoding='utf-8')
        (self.repo/'rvc/train/train.py').write_text('from data_utils import MARKER\nif False:\n load_checkpoint()\n latest_checkpoint_path()\nprint(MARKER)\n',encoding='utf-8')
        helper=self.root/'helper';helper.mkdir()
        (helper/'rvc_resume.py').write_text('def install_guard(path): pass\n',encoding='utf-8')
        runner,_=r.guarded_runner(self.repo,self.root/'job',self.root/'backup',{'complete':True},helper,engine='applio')
        result=subprocess.run([sys.executable,str(runner)],cwd=self.repo,capture_output=True,text=True,timeout=10)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn('native_import_ok',result.stdout)

    def test_index_writer_uses_vector_io_and_writes_unicode_target(self):
        import subprocess
        (self.repo/'rvc/train/process').mkdir(parents=True)
        (self.repo/'rvc/train/process/extract_index.py').write_text('import faiss,sys\nfrom pathlib import Path\nfaiss.write_index(object(),str(Path(sys.argv[1])/"voice.index"))\n',encoding='utf-8')
        output=self.root/'中文索引';output.mkdir()
        job=self.root/'index-job'
        runner=a.index_runner(self.repo,job,output,'voice')
        (job/'faiss.py').write_text('class VectorIOWriter:\n pass\ndef write_index(index,target,*args):\n assert isinstance(target,VectorIOWriter)\n target.data=b"fixture-index"\nclass Array:\n def tobytes(self): return b"fixture-index"\ndef vector_to_array(value): return Array()\ndef serialize_index(index): raise AssertionError("must avoid patched write recursion")\n',encoding='utf-8')
        result=subprocess.run([sys.executable,str(runner),str(output),'Auto'],cwd=self.repo,capture_output=True,text=True,timeout=10)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual((output/'voice.index').read_bytes(),b'fixture-index')


if __name__=='__main__': unittest.main()
