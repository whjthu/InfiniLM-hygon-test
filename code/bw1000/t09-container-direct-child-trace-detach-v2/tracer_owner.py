"""A directed detach for a single unreaped, directly owned child."""
import os
from pathlib import Path
import signal
import threading


def need(flag,code):
    if not flag:
        raise ValueError(code)


def boot_id():
    return Path('/proc/sys/kernel/random/boot_id').read_text().strip()


def native_thread_count():
    return len(list(Path('/proc/self/task').iterdir()))


def child_ids():
    tid=os.getpid()
    return (Path('/proc/self/task')/str(tid)/'children').read_text().split()


def check_parent():
    need(callable(os.kill) and hasattr(signal,'SIGINT') and hasattr(signal,'SIGCHLD')
         and signal.getsignal(signal.SIGCHLD)==signal.SIG_DFL
         and threading.active_count()==native_thread_count()==1,
         'DIRECT_CHILD_OWNER_SINGLE_THREAD_DEFAULT_SIGCHLD_REQUIRED')


def resolved_executable(path):
    actual=path.resolve(strict=True)
    state=actual.stat()
    return str(actual),state.st_dev,state.st_ino


def child_executable(pid):
    executable=Path('/proc')/str(pid)/'exe'
    actual=os.readlink(executable)
    state=executable.stat()
    return actual,state.st_dev,state.st_ino


class Claim:
    def __init__(self,path):
        # Assert the contract before Popen and before any scope child can exist.
        check_parent()
        need(not child_ids(),'NO_OTHER_CHILD_OR_UNREAPED_PROCESS_BEFORE_TRACER')
        self.expected_executable=resolved_executable(path)
        self.boot=boot_id()
        need(bool(self.boot),'EXACT_CURRENT_BOOT_ID_REQUIRED')
        self.ppid=os.getpid()
        self.mode=('PIDFD' if callable(getattr(os,'pidfd_open',None))
                   and callable(getattr(signal,'pidfd_send_signal',None)) else 'DIRECT_UNREAPED_CHILD')
        self.anchor=None
        self.attempted=False
        self.fd=None

    def sample(self,child,proc):
        check_parent()
        pid=child.pid
        need(type(pid) is int and pid>0 and pid!=self.ppid and child.returncode is None
             and self.ppid==os.getpid() and self.boot==boot_id() and child_ids()==[str(pid)],
             'EXACT_POSITIVE_UNREAPED_DIRECT_CHILD_ON_SAME_BOOT')
        state=proc(pid)
        need((state['tgid'],state['tid'],state['ppid'])==(pid,pid,self.ppid)
             and type(state['startticks']) is int and state['startticks']>0
             and state['state'] not in ('Z','X','x') and child_executable(pid)==self.expected_executable,
             'EXACT_LIVE_TRACER_STARTTICKS_PPID_AND_EXECUTABLE')
        return pid,state['startticks']

    def open(self,child,proc):
        need(self.anchor is None,'EXACTLY_ONE_TRACER_IDENTITY_ANCHOR')
        self.anchor=self.sample(child,proc)
        if self.mode=='PIDFD':
            # A failed advertised pidfd syscall is not a reason to downgrade.
            self.fd=os.pidfd_open(self.anchor[0])
        return dict(mode=self.mode,pid=self.anchor[0],startticks=self.anchor[1],ppid=self.ppid,
                    boot_id=self.boot,exe_device_inode=list(self.expected_executable[1:]))

    def interrupt(self,child,proc):
        need(self.anchor is not None and self.sample(child,proc)==self.anchor and not self.attempted,
             'EXACT_SAME_UNREAPED_TRACER_BEFORE_ONLY_SIGNAL')
        self.attempted=True
        if self.mode=='PIDFD':
            need(self.fd is not None,'EXACT_LIVE_PIDFD_REQUIRED')
            signal.pidfd_send_signal(self.fd,signal.SIGINT)
        else:
            # A direct child cannot be recycled until this sole parent reaps it.
            # No wait/poll or registered SIGCHLD handler can occur before the send.
            os.kill(self.anchor[0],signal.SIGINT)

    def close(self):
        if self.fd is not None:
            os.close(self.fd)
            self.fd=None
