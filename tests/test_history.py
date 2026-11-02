import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from nulvoiceagent import brain, dispatch, history

MSG = ("Can you tell three D model Claude uh to um we need to fix the texturing on "
       "the trees. Merge the root blocks and unwrap the UVs.")


class HistoryTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.p = patch.object(history, "PATH", Path(self.dir.name) / "h.jsonl")
        self.p.start()

    def tearDown(self):
        self.p.stop()
        self.dir.cleanup()

    def test_record_and_hint_excludes_current_utterance(self):
        a = history.record("you", MSG)
        history.record("route", "I don't have a pane named three D model Claude uh.")
        history.record("you", "forward that to 3D model Claude")
        h = history.hint()
        self.assertIn(f"#{a} ", h)
        self.assertIn("router: I don't have a pane", h)
        self.assertNotIn("forward that to 3D model Claude", h)

    def test_resolve_ref_is_verbatim_minus_routing_prefix(self):
        a = history.record("you", MSG)
        self.assertEqual(history.resolve_ref(f"#{a}"),
                         "We need to fix the texturing on the trees. Merge the root "
                         "blocks and unwrap the UVs.")
        self.assertEqual(history.resolve_ref("just do the thing"), "just do the thing")
        self.assertEqual(history.resolve_ref("#9999"), "#9999")

    def test_old_entries_age_out(self):
        history.record("you", "ancient")
        rows = history._read(); rows[0]["ts"] = time.time() - 99999
        history.PATH.write_text("".join(__import__("json").dumps(r) + "\n" for r in rows))
        self.assertEqual(history.recent(), [])

    def test_chat_act_substitutes_history_reference(self):
        a = history.record("you", MSG)
        history.record("you", "send that message to 3D model Claude")
        reply_raw = f"Sending it over. [[ask: 3D model Claude | #{a}]]"
        with patch.object(brain.dispatch, "load", return_value={"3D model Claude": {}}), \
             patch.object(brain, "_style_hint", return_value=""), \
             patch.object(brain, "_chat_dispatch", return_value=reply_raw) as run:
            reply, action, ask = brain.chat_act("send that message to 3D model Claude", [])
        self.assertIn(f"#{a} ", run.call_args.args[0])
        self.assertEqual(ask[0], "3D model Claude")
        self.assertTrue(ask[1].startswith("We need to fix the texturing"))


class FillerTests(unittest.TestCase):
    def test_trailing_filler_does_not_break_pane_match(self):
        t = {"3D model Claude": {}, "Krita Claude": {}}
        self.assertEqual(dispatch._match_key("three D model Claude uh", t), "3D model Claude")
        self.assertEqual(dispatch._match_key("Krita Claude um", t), "Krita Claude")


if __name__ == "__main__":
    unittest.main()
