"""GPT6-Astra: reject fabricated timing/partial traces; preserve host evidence."""
import importlib.util
import sys
sys.dont_write_bytecode = True
import tempfile
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location("trace", Path(__file__).parents[1]/"GPU_Analyzer/analyze_workload_trace.py")
trace = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trace)


class TraceTests(unittest.TestCase):
    def summarize(self, rows, meta="committed=1,overflow=0,drained=1"):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"trace.csv"
            path.write_text("# version=1,batch=32,"+meta+",wall_us=100\n"
                            "kind,stage,name,bytes,host_us,start_us,end_us,duration_us,api_ok,event_ok,timeline_ok\n"+rows)
            return trace.analyze(path)

    def test_valid_units(self):
        result = self.summarize("memset,memset,hipMemset,1000,10,0,20,20,1,1,1\n"
                                "kernel,backward,k,0,1,25,30,5,1,1,1\n"
                                "sync,backward,sync,0,7,-1,-1,-1,1,1,0\n")
        self.assertEqual(result["groups"]["memset/memset/all"]["event_envelope_byte_rate_GBs"], .05)
        self.assertEqual(result["host_sync_us"], 7)
        self.assertEqual(result["kernel_launches"], 1)
        self.assertIsNone(result["gpu_idle_us"])

    def test_bad_event_only_invalidates_its_group_and_full_timeline(self):
        result = self.summarize("memset,memset,hipMemset,1000,10,0,20,20,1,1,1\n"
                                "kernel,backward,k,0,1,30,29,-1,1,0,0\n")
        self.assertEqual(result["invalid_device_events"], 1)
        self.assertEqual(result["kernel_event_count"], 0)
        self.assertIsNone(result["kernel_envelopes"])
        memset = result["groups"]["memset/memset/all"]
        self.assertTrue(memset["event_complete"])
        self.assertEqual(memset["event_coverage"], 1.0)
        self.assertEqual(memset["event_envelope_byte_rate_GBs"], .05)
        self.assertFalse(result["timeline_valid"])
        self.assertEqual(result["groups"]["memset/memset/all"]["host_api_byte_rate_GBs"], .1)

    def test_partial_group_reports_valid_subset_and_coverage(self):
        result = self.summarize("memset,memset,hipMemset,1000,10,0,10,10,1,1,1\n"
                                "memset,memset,hipMemset,3000,20,30,29,-1,1,0,0\n"
                                "kernel,backward,k,0,1,40,45,5,1,1,1\n")
        memset = result["groups"]["memset/memset/all"]
        self.assertFalse(memset["event_complete"])
        self.assertEqual(memset["event_coverage"], .5)
        self.assertIsNone(memset["event_envelope_byte_rate_GBs"])
        self.assertEqual(memset["valid_event_bytes"], 1000)
        self.assertEqual(memset["valid_event_byte_rate_GBs"], .1)
        self.assertEqual(memset["valid_event_duration"]["count"], 1)
        self.assertEqual(result["kernel_event_coverage"], 1.0)
        self.assertEqual(result["kernel_envelopes"]["count"], 1)

    def test_incomplete(self):
        for meta in ("committed=0,overflow=0,drained=1", "committed=1,overflow=1,drained=1"):
            with self.assertRaises(ValueError):
                self.summarize("", meta)


if __name__ == "__main__":
    unittest.main()
