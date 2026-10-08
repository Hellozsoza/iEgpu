"""Host tests: python3 -m unittest discover -s tests/linux -v"""

import argparse
import importlib.util
from pathlib import Path
import socket
import subprocess
import threading
import unittest
from unittest.mock import Mock, patch

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/serve-linux.py"
SPEC = importlib.util.spec_from_file_location("linux_worker", SCRIPT)
worker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(worker)


class WorkerTests(unittest.TestCase):
    def test_no_network_device_discovery_and_ambiguous_phone_fails(self):
        result = subprocess.CompletedProcess([], 0, "one\ntwo\n", "")
        with patch.object(worker, "require_tool", return_value="idevice_id"), \
                patch.object(worker.subprocess, "run", return_value=result) as run:
            with self.assertRaisesRegex(worker.WorkerError, "Multiple"):
                worker.choose_phone()
            self.assertEqual(worker.choose_phone("two"), "two")
            self.assertEqual(run.call_args.args[0], ["idevice_id", "-l"])
            with self.assertRaisesRegex(worker.WorkerError, "not connected"):
                worker.choose_phone("absent")

    def test_old_cache_enabled_and_non_worker_apps_are_rejected(self):
        for info in ({}, {"rpc_cache_enabled": True}, {"rpc_cache_enabled": 0},
                     {"rpc_cache_enabled": False, "rpc_only": False}):
            with self.subTest(info=info), self.assertRaises(worker.WorkerError):
                worker.require_ram_worker(info)
        worker.require_ram_worker({"rpc_cache_enabled": False, "rpc_only": True})

    def test_phone_control_handles_fragmented_json_and_early_eof(self):
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            listener.listen()
            port = listener.getsockname()[1]
            seen = []

            def phone():
                for reply in (b'{"rpc_cache_enabled": false}\n', b'{"unfinished":'):
                    with listener.accept()[0] as conn:
                        seen.append(conn.recv(64))
                        for byte in reply:
                            conn.sendall(bytes([byte]))

            thread = threading.Thread(target=phone)
            thread.start()
            try:
                self.assertEqual(worker.control_memory(port), {"rpc_cache_enabled": False})
                with self.assertRaisesRegex(worker.WorkerError, "Invalid reply"):
                    worker.control_memory(port)
            finally:
                thread.join(timeout=3)
            self.assertEqual(seen, [b"mem\n", b"mem\n"])

    def test_remote_device_is_discovered_through_real_engine_cli(self):
        result = subprocess.CompletedProcess([], 0, "Available devices:\n  RPC0: phone (4096 MiB)\n", "")
        with patch.object(worker.subprocess, "run", return_value=result) as run:
            self.assertEqual(worker.remote_device(Path("server"), "localhost:123"), "RPC0")
            self.assertEqual(run.call_args.args[0], ["server", "--rpc", "localhost:123", "--list-devices"])
            result.stdout += "  RPC1: unwanted second device\n"
            with self.assertRaisesRegex(worker.WorkerError, "Expected one"):
                worker.remote_device(Path("server"), "localhost:123")

    def test_weights_and_embeddings_are_on_phone_by_default(self):
        args = argparse.Namespace(server=Path("server"), model=Path("model with spaces.gguf"),
                                  gpu_layers=999, ctx=2048, port=8080)
        command = worker.server_command(args, "localhost:123", "RPC0")
        self.assertEqual(command[command.index("--model") + 1], "model with spaces.gguf")
        self.assertEqual(command[command.index("--device") + 1], "RPC0")
        self.assertEqual(command[command.index("--override-tensor") + 1], ".=RPC0[localhost:123]")
        self.assertEqual(command[command.index("--fit") + 1], "off")
        self.assertEqual(command[command.index("--host") + 1], "127.0.0.1")
        args.gpu_layers = 8
        self.assertNotIn("--override-tensor", worker.server_command(args, "localhost:123", "RPC0"))

    def test_tunnel_is_usb_only_and_cleaned_up_on_cache_rejection(self):
        process = Mock()
        process.poll.return_value = None
        with patch.object(worker, "require_tool", return_value="iproxy"), \
                patch.object(worker.subprocess, "Popen", return_value=process) as popen, \
                patch.object(worker, "control_memory", return_value={"rpc_cache_enabled": True}):
            with self.assertRaisesRegex(worker.WorkerError, "caching"):
                with worker.usb_tunnel("selected-phone"):
                    self.fail("Unsafe app must not reach model loading")
            command = popen.call_args.args[0]
            self.assertEqual(command[:7], ["iproxy", "-l", "-u", "selected-phone", "-s", "127.0.0.1", command[6]])
            self.assertTrue(command[6].endswith(":50052"))
            self.assertTrue(command[7].endswith(":50061"))
            self.assertNotIn("-n", command)
            process.terminate.assert_called_once()
            process.wait.assert_called_once()

    def test_child_exit_status_and_disconnect_cleanup(self):
        child = Mock(returncode=7)
        child.poll.return_value = 7
        tunnel = Mock()
        tunnel.poll.return_value = None
        with patch.object(worker.subprocess, "Popen", return_value=child) as popen:
            self.assertEqual(worker.run_server(["server"], tunnel), 7)
            self.assertEqual(popen.call_args.kwargs["env"]["LLAMA_LAZY_EMBD"], "0")
            child.poll.return_value = None
            tunnel.poll.return_value = 1
            with self.assertRaisesRegex(worker.WorkerError, "tunnel stopped"):
                worker.run_server(["server"], tunnel)
            child.terminate.assert_called_once()

    def test_stuck_children_are_killed(self):
        process = Mock()
        process.poll.return_value = None
        process.wait.side_effect = [subprocess.TimeoutExpired("iproxy", 5), 0]
        worker.stop_process(process)
        process.kill.assert_called_once()


if __name__ == "__main__":
    unittest.main()
