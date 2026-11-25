"""Retire only verified empty directories left by this request's provider."""

import os
import re
import stat


_PROVIDER_TEMP = re.compile(r"pymp-[a-z0-9_]{8}\Z")
_FLAGS = lambda: os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW|os.O_CLOEXEC


def _exact(condition, code):
    if not condition:
        raise ValueError(code)


def _identity(info):
    return info.st_dev,info.st_ino,info.st_uid,info.st_mode


def retire_empty_pool_paths(root, *, initially_marker_only, children_reaped, report,
                            required_marker=".scope"):
    report.update(initially_marker_only=initially_marker_only,children_reaped=children_reaped,
                  state="REFUSED",before=None,after=None,removed=0)
    _exact(initially_marker_only is True and children_reaped is True,"G42_RETIREMENT_LINEAGE_OR_CHILDREN_UNPROVEN")
    uid=os.getuid()
    fd=os.open(root,_FLAGS())
    opened=[]
    try:
        inode=os.fstat(fd)
        def exact_root():
            return root.resolve(strict=True)==root and _identity(root.lstat())==_identity(inode)
        _exact(stat.S_ISDIR(inode.st_mode) and inode.st_uid==uid and stat.S_IMODE(inode.st_mode)==0o700
               and exact_root(),"G42_RETIREMENT_PRIVATE_ROOT_IDENTITY")

        def inventory():
            names=os.listdir(fd)
            _exact(len(names)<=128,"G42_RETIREMENT_INVENTORY_BOUND")
            marker=False
            unexpected=0
            directories=[]
            for name in sorted(names):
                info=os.stat(name,dir_fd=fd,follow_symlinks=False)
                if name==required_marker:
                    marker=(stat.S_ISREG(info.st_mode) and info.st_uid==uid and
                            stat.S_IMODE(info.st_mode)==0o600 and info.st_nlink==1)
                elif (_PROVIDER_TEMP.fullmatch(name) and stat.S_ISDIR(info.st_mode)
                        and info.st_uid==uid and stat.S_IMODE(info.st_mode)==0o700
                        and info.st_dev==inode.st_dev):
                    directories.append((name,info))
                else:
                    unexpected+=1
            summary={"marker_valid":marker,"candidate_directories":len(directories),"unexpected":unexpected}
            return summary,directories

        before,candidates=inventory()
        report["before"]=before
        try:
            _exact(before["marker_valid"] and before["unexpected"]==0,"G42_RETIREMENT_UNEXPECTED_MEMBER")
            # Inspect all candidates before removing any. Open relative to the
            # same bound root and hold each exact directory identity.
            for name,prior in candidates:
                child=os.open(name,_FLAGS(),dir_fd=fd)
                opened.append((name,child,prior))
                _exact(_identity(os.fstat(child))==_identity(prior) and not os.listdir(child),
                       "G42_RETIREMENT_NONEMPTY_OR_CHANGED_MEMBER")
            for name,child,prior in opened:
                rechecked=os.stat(name,dir_fd=fd,follow_symlinks=False)
                _exact(exact_root() and _identity(rechecked)==_identity(prior) and not os.listdir(child),
                       "G42_RETIREMENT_MEMBER_CHANGED_BEFORE_RMDIR")
                os.rmdir(name,dir_fd=fd)
                report["removed"]+=1
        finally:
            if exact_root():
                report["after"],_=inventory()
        _exact(report["after"]=={"marker_valid":True,"candidate_directories":0,"unexpected":0},
               "G42_RETIREMENT_NOT_CLEAN")
        report["state"]="MARKER_ONLY"
    finally:
        for _,child,_ in opened:
            os.close(child)
        os.close(fd)
