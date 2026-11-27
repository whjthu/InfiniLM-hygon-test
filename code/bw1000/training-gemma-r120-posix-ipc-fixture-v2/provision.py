"""Provision one provider temp directory. Retire only the one this process created."""

import ast
from hashlib import sha256
import inspect
import os
from pathlib import Path
import re
import stat
import tempfile
from textwrap import dedent


OLD_REQUEST="35345a3f196b1d04ec1c4157e9d1c0ce"


def need(ok,reason):
    if not ok:
        raise ValueError(reason)


def physical(meta):
    return (meta.st_dev,meta.st_ino,meta.st_uid,stat.S_IMODE(meta.st_mode))


def mounting(fd):
    text=(Path("/proc/self/fdinfo")/str(fd)).read_text()
    values=re.findall(r"^mnt_id:\s*([0-9]+)$",text,re.M)
    need(len(values)==1 and int(values[0])>0,"EXACT_COMPUTE_MOUNT_ID")
    return int(values[0])


def process_stamp():
    statline=Path("/proc/self/stat").read_text()
    return int(statline.rsplit(") ",1)[1].split()[19])


def route_semantics(util,process,connection):
    def syntax(fn):
        return ast.parse(dedent(inspect.getsource(fn))).body[0]
    getter=syntax(util.get_temp_dir)
    ctor=syntax(process.BaseProcess.__init__)
    addresses=syntax(connection.arbitrary_address)
    opts=tuple(ast.unparse(stmt) for stmt in getter.body[:1]+getter.body[-1:])
    supported=(opts==("tempdir = process.current_process()._config.get('tempdir')","return tempdir")
               and len(getter.body)==3 and isinstance(getter.body[1],ast.If)
               and ast.unparse(getter.body[1].test)=="tempdir is None" and not getter.body[1].orelse
               and any(isinstance(n,ast.Assign) and ast.unparse(n)=="self._config = _current_process._config.copy()"
                       for n in ctor.body)
               and any(isinstance(n,ast.Call) and ast.unparse(n.func)=="tempfile.mktemp"
                       and any(k.arg=="dir" and ast.unparse(k.value)=="util.get_temp_dir()" for k in n.keywords)
                       for n in ast.walk(addresses)))
    need(supported,"NATIVE_TEMP_CACHE_ROUTE_UNPROVEN")
    return dict(getter_cache_short_circuit=True,child_config_copy=True,unix_listener_uses_cache=True)


def verify_provider(provider,util,process,connection,proof):
    """Bind deployed source bytes and callable bodies before touching the process cache."""
    actual={}
    for key,module,methods in (
        ("util",util,(("get_temp_dir",util.get_temp_dir),("_remove_temp_dir",util._remove_temp_dir),)),
        ("process",process,(("BaseProcess.__init__",process.BaseProcess.__init__),)),
        ("connection",connection,(("arbitrary_address",connection.arbitrary_address),)),
    ):
        item=proof[key]
        path=Path(module.__file__).resolve(strict=True)
        need(path.as_posix()==item["path"] and path.is_file(),"EXACT_PROVIDER_CANONICAL_PATH")
        with path.open("rb") as stream:
            raw=stream.read(256*1024+1)
        pin=sha256(raw).hexdigest()
        need(0<len(raw)==item["bytes"]<=256*1024 and pin==item["SHA256"],"EXACT_PROVIDER_FILE_PIN")
        bodies={}
        for name,fn in methods:
            source,lineno=inspect.getsourcelines(fn)
            tree=ast.unparse(ast.parse(dedent("".join(source))))
            body=sha256(tree.encode()).hexdigest()
            need(Path(inspect.getfile(fn)).resolve()==path
                 and item["methods"][name]==dict(line=lineno,SHA256=body),"EXACT_PROVIDER_METHOD_BODY")
            bodies[name]=dict(line=lineno,SHA256=body)
        actual[key]=dict(bytes=len(raw),SHA256=pin,methods=bodies)
    need(provider.current_process is process.current_process,"EXACT_IMPORTED_CURRENT_PROCESS")
    return actual | {"semantics":route_semantics(util,process,connection)}


class ProvisionedPool:
    """Use the upstream cached-directory branch, leaving its socket lifecycle unchanged."""

    def __init__(self,root,request,*,posix=True):
        self.root=Path(root)
        self.request=request
        self.posix=posix
        self.state="UNCREATED"
        self.root_fd=self.child_fd=None
        self.provider_config=None
        self.provider=None
        self.attached=False
        self.removals=0

    def create(self, *, peers=()):
        need(self.state=="UNCREATED" and re.fullmatch("[0-9a-f]{32}",self.request) and self.request!=OLD_REQUEST
             and self.root.resolve(strict=True)==self.root and set(os.listdir(self.root))==self.expected(peers),
             "NEW_MARKER_ONLY_SCOPE_REQUIRED")
        marker=self.root/".scope"
        before=marker.lstat()
        need(stat.S_ISREG(before.st_mode) and before.st_nlink==1 and marker.read_bytes()==self.request.encode()+b"\n",
             "EXACT_SCOPE_MARKER")
        self.marker=physical(before)+(before.st_mtime_ns,before.st_ctime_ns)
        self.parent=physical(self.root.lstat())
        if self.posix:
            need(os.name=="posix" and self.parent[2:]==(os.getuid(),0o700)
                 and self.marker[2:4]==(os.getuid(),0o600),"PRIVATE_UID_AND_MODES")
            self.flags=os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW|os.O_CLOEXEC
            self.root_fd=os.open(self.root,self.flags)
            need(physical(os.fstat(self.root_fd))==self.parent,"EXACT_ROOT_FD")
            self.mount=mounting(self.root_fd)
            self.creator_startticks=process_stamp()
        else:
            self.mount=self.creator_startticks=None
        self.creator_pid=os.getpid()
        self.child=Path(tempfile.mkdtemp(prefix="pymp-",dir=self.root))
        self.identity=physical(self.child.lstat())
        self.verify()
        if self.posix:
            self.child_fd=os.open(self.child.name,self.flags,dir_fd=self.root_fd)
            need(physical(os.fstat(self.child_fd))==self.identity and mounting(self.child_fd)==self.mount,
                 "EXACT_CREATED_CHILD_FD_AND_MOUNT")
        self.state="CREATED"
        return self.evidence()

    def expected(self, peers):
        names = {".scope"}
        for peer in peers:
            need(peer is not self and peer.root == self.root and peer.request == self.request
                 and peer.state == "CREATED" and peer.child.name not in names
                 and peer.creator_pid == os.getpid(), "EXACT_COHORT_PEER")
            peer.verify()
            names.add(peer.child.name)
        return names

    def verify(self):
        marker=(self.root/".scope").lstat()
        metadata=self.child.lstat()
        need(physical(self.root.lstat())==self.parent and
             physical(marker)+(marker.st_mtime_ns,marker.st_ctime_ns)==self.marker and
             stat.S_ISDIR(metadata.st_mode) and self.child.parent==self.root and
             physical(metadata)==self.identity and self.identity[0]==self.parent[0]
             and self.identity[2]==self.parent[2] and (not self.posix or self.identity[3]==0o700),
             "CREATED_DIRECTORY_OR_MARKER_REPLACED")

    def attach(self,provider,util,process,connection,proof):
        need(self.state=="CREATED" and not self.attached and os.getpid()==self.creator_pid,
             "EXACT_PARENT_OWNER_ONLY")
        self.verify()
        verify_provider(provider,util,process,connection,proof)
        config=provider.current_process()._config
        need(type(config) is dict and config.get("tempdir") is None,"PROVIDER_HAS_AN_EXISTING_DIRECTORY")
        self.had_key="tempdir" in config
        config["tempdir"]=str(self.child)
        self.provider_config=config
        self.provider=provider
        self.attached=True
        need(util.get_temp_dir()==str(self.child),"NATIVE_PROVIDER_DID_NOT_USE_CREATED_DIRECTORY")
        return self.evidence()

    def preflight(self,*,children_reaped,peers=()):
        need(self.state=="CREATED" and self.attached and children_reaped is True and os.getpid()==self.creator_pid
             and (not self.posix or process_stamp()==self.creator_startticks),"PROVEN_ORIGINAL_PARENT_AND_REAP_REQUIRED")
        need(not self.provider.active_children(),"ACTIVE_NATIVE_PROVIDER_CHILDREN")
        if self.posix:
            task=Path("/proc/self/task")/str(self.creator_pid)/"children"
            need(not task.read_text().split(),"ACTIVE_DIRECT_SCOPE_CHILDREN")
        self.verify()
        need(self.provider_config.get("tempdir")==str(self.child),"UNEXPECTED_PROVIDER_CACHE_MUTATION")
        need(set(os.listdir(self.root))==self.expected(peers)|{self.child.name},"UNREGISTERED_RESIDUE_RETAINED")
        if self.posix:
            need(physical(os.fstat(self.root_fd))==self.parent and physical(os.fstat(self.child_fd))==self.identity and
                 mounting(self.root_fd)==mounting(self.child_fd)==self.mount and not os.listdir(self.child_fd),
                 "CREATED_DIR_NONEMPTY_OR_FD_CHANGED")
            link=os.stat(self.child.name,dir_fd=self.root_fd,follow_symlinks=False)
            need(stat.S_ISDIR(link.st_mode) and physical(link)==self.identity,"FINAL_DIRECTORY_LOOKUP_CHANGED")
        else:
            need(not os.listdir(self.child),"CREATED_DIR_NONEMPTY_OR_FD_CHANGED")
            self.verify()

    def finish(self,*,children_reaped,peers=()):
        self.preflight(children_reaped=children_reaped,peers=peers)
        if self.posix:
            os.rmdir(self.child.name,dir_fd=self.root_fd)
        else:
            self.child.rmdir()
        self.removals=1
        self.restore()
        need(set(os.listdir(self.root))==self.expected(peers),"FINAL_DIRECTORY_DIVERGED")
        self.state="RETIRED"
        return self.evidence()

    def restore(self):
        if self.attached:
            need(self.provider_config.get("tempdir")==str(self.child),"UNEXPECTED_PROVIDER_CACHE_MUTATION")
            if self.had_key:
                self.provider_config["tempdir"]=None
            else:
                self.provider_config.pop("tempdir")
            self.attached=False

    def close(self):
        try:
            self.restore()
        finally:
            for handle in (self.child_fd,self.root_fd):
                if handle is not None:
                    os.close(handle)
            self.child_fd=self.root_fd=None

    def evidence(self):
        return dict(state=self.state,creator_pid=self.creator_pid,creator_startticks=self.creator_startticks,
                    inode=self.identity[1],uid=self.identity[2],device=self.identity[0],mount_id=self.mount,
                    name_SHA256=sha256(os.fsencode(self.child.name)).hexdigest(),
                    recorded_creation=True,provider_cache_bound=self.attached,
                    directory_removals=self.removals)
