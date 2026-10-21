import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from nulvoiceagent import brain, config, distill


class RoutingTests(unittest.TestCase):
    def test_voice_skill_is_supplied_with_current_actions_and_panes(self):
        with patch.object(brain.dispatch, 'load', return_value={'Krita Codex': {}}), \
             patch.object(brain, '_style_hint', return_value=''), \
             patch.object(brain, '_chat_dispatch', return_value='Opening. [[do: krita.launch]]') as run:
            reply, action, ask = brain.chat_act('open Krita', ['krita.launch'])
        prompt = run.call_args.args[0]
        self.assertIn(brain._command_skill_hint(), prompt)
        self.assertIn('Krita Codex', prompt)
        self.assertIn('krita.launch', prompt)
        self.assertEqual((reply, action, ask), ('Opening.', 'krita.launch', None))

    def test_passive_speech_does_not_receive_command_skill(self):
        with patch.object(brain, '_style_hint', return_value=''), \
             patch.object(brain, '_chat_dispatch', return_value='PASS') as run:
            brain.eavesdrop('clip that')
        self.assertNotIn('Voice command skill:', run.call_args.args[0])

    def test_missing_command_skill_preserves_existing_routing(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(config, 'REPO_DIR', Path(directory)):
            self.assertEqual(brain._command_skill_hint(), '')

    def test_defaults(self):
        self.assertEqual((config.CHAT_BRAIN, config.CMD_BRAIN), ('codex', 'codex'))
        self.assertEqual((config.CHAT_FALLBACK, config.CMD_FALLBACK), ('claude', 'claude'))

    def test_chat_and_commands_try_codex_then_claude(self):
        for method, registry, args in [
            (brain.chat, brain._CHAT, ('hello',)),
            (brain.classify, brain._CLASSIFY, ('hello', ['hello'])),
        ]:
            calls = []
            def codex(*args):
                calls.append('codex')
                raise brain.BrainUnavailable('unavailable')
            def claude(*args):
                calls.append('claude')
                return 'answer'
            with tempfile.TemporaryDirectory() as directory, \
                 patch.object(config, 'BRAIN_COOLDOWN_FILE', Path(directory) / 'cooldown'), \
                 patch.object(config, 'CHAT_SESSION_FILE', Path(directory) / 'session'), \
                 patch.object(brain, '_log'), \
                 patch.dict(registry, codex=codex, claude=claude):
                self.assertEqual(method(*args), 'answer')
                self.assertEqual(calls, ['codex', 'claude'])

    def test_processing_codex_success_does_not_call_claude(self):
        def run(cmd, **kwargs):
            self.assertEqual(cmd[0], 'codex')
            self.assertIn('read-only', cmd)
            Path(cmd[cmd.index('-o') + 1]).write_text('answer')
            return subprocess.CompletedProcess(cmd, 0, '', '')
        with patch.object(brain.subprocess, 'run', side_effect=run) as mocked:
            self.assertEqual(brain.complete('content', 'instruction'), 'answer')
            self.assertEqual(mocked.call_count, 1)

    def test_processing_falls_back_on_failure_or_missing_cli(self):
        for first in [subprocess.CompletedProcess([], 1, '', 'failed'), FileNotFoundError()]:
            with patch.object(brain, '_log'), patch.object(brain.subprocess, 'run', side_effect=[
                first, subprocess.CompletedProcess([], 0, '{"result":"fallback"}', '')
            ]) as mocked:
                self.assertEqual(brain.complete('content', 'instruction'), 'fallback')
                self.assertEqual([c.args[0][0] for c in mocked.call_args_list], ['codex', 'claude'])

    def test_all_text_processing_uses_shared_routing(self):
        with patch.object(brain, 'complete', return_value='answer') as mocked:
            self.assertEqual(brain.summarize('completion'), 'answer')
            self.assertEqual(brain.condense('page'), 'answer')
            self.assertEqual(distill._ask_model('memory'), 'answer')
            self.assertEqual(mocked.call_count, 3)

    def test_processing_rejects_claude_limit_notice(self):
        with patch.object(brain, '_log'), patch.object(brain.subprocess, 'run', side_effect=[
            subprocess.CompletedProcess([], 1, '', 'failed'),
            subprocess.CompletedProcess([], 0, '{"result":"You\'ve hit your monthly spend limit"}', '')
        ]):
            with self.assertRaises(brain.BrainUnavailable):
                brain.complete('content', 'instruction')


if __name__ == '__main__':
    unittest.main()
