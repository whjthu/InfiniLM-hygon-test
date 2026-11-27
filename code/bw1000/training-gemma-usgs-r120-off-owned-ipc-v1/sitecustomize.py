"""Activate custody in the exact rank process before the Swift script imports."""

import os
import sys

if os.environ.get("GEMMA_IPC_OPERATION") and sys.argv and sys.argv[0] == os.environ.get("GEMMA_IPC_RANK_ENTRY"):
    from rank_ipc import rank_on_entry

    rank_on_entry()
