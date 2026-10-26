import asyncio
import ctypes
import os
import subprocess
import sys
import unittest

from personal_agent.media import terminate_tree


@unittest.skipUnless(os.name == "nt", "Windows process-tree cleanup")
class ProcessCleanupTests(unittest.IsolatedAsyncioTestCase):
    async def test_only_owned_dummy_process_tree_is_terminated(self):
        code = "import subprocess,time,sys; p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); print(p.pid,flush=True); time.sleep(60)"
        process = await asyncio.create_subprocess_exec(
            sys.executable, "-u", "-c", code, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL, creationflags=subprocess.CREATE_NO_WINDOW,
        )
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = (ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong)
        kernel.OpenProcess.restype = ctypes.c_void_p
        kernel.WaitForSingleObject.argtypes = (ctypes.c_void_p, ctypes.c_ulong)
        kernel.WaitForSingleObject.restype = ctypes.c_ulong
        kernel.CloseHandle.argtypes = (ctypes.c_void_p,)
        kernel.CloseHandle.restype = ctypes.c_int
        child_handle = None
        try:
            child_id = int(await asyncio.wait_for(process.stdout.readline(), 5))
            child_handle = kernel.OpenProcess(0x00100000, False, child_id)
            self.assertTrue(child_handle)
            self.assertEqual(kernel.WaitForSingleObject(child_handle, 0), 258)
            await terminate_tree(process)
            self.assertIsNotNone(process.returncode)
            self.assertEqual(kernel.WaitForSingleObject(child_handle, 3000), 0)
        finally:
            await terminate_tree(process)
            if child_handle:
                kernel.CloseHandle(child_handle)
