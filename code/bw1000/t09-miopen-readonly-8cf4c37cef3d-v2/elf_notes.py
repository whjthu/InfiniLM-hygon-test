"""Cross-check readelf's GNU Build ID projection against actual ELF64 note records."""
from hashlib import sha256
import os
import re
import struct


class NoteError(ValueError):
    pass


def require(ok,code):
    if not ok:
        raise NoteError(code)


def bounded_read(stream,offset,size,extent):
    require(0<=offset<=extent and 0<=size<=extent-offset,'MALFORMED_ELF_NOTE_BOUNDS')
    stream.seek(offset)
    raw=stream.read(size)
    require(len(raw)==size,'TRUNCATED_ELF_NOTE_STRUCTURE')
    return raw


def scan_range(raw,base,order):
    cursor=0
    matches={}
    count=0
    align=lambda number:(number+3)&~3
    while cursor<len(raw):
        if len(raw)-cursor<12:
            require(not any(raw[cursor:]),'MALFORMED_ELF_NOTE_TRAILER')
            break
        count+=1
        require(count<=16384,'ELF_NOTE_RECORD_LIMIT')
        names,descs,kind=struct.unpack_from(order+'III',raw,cursor)
        begin=cursor+12
        name_end=begin+names
        desc_begin=begin+align(names)
        desc_end=desc_begin+descs
        nxt=desc_begin+align(descs)
        require(nxt<=len(raw),'MALFORMED_ELF_NOTE_RECORD')
        name=raw[begin:name_end]
        desc=raw[desc_begin:desc_end]
        if name.rstrip(b'\0')==b'GNU' and kind==3:
            require(name==b'GNU\0' and 4<=len(desc)<=64,'MALFORMED_GNU_BUILD_ID_NOTE')
            matches[base+cursor]=desc.hex()
        cursor=nxt
    return matches


def file_notes(stream,extent):
    header=bounded_read(stream,0,64,extent)
    require(header[:5]==b'\x7fELF\x02' and header[5] in (1,2) and header[6]==1,'UNSUPPORTED_ELF_NOTE_LAYOUT')
    order='<' if header[5]==1 else '>'
    fields=struct.unpack(order+'16sHHIQQQIHHHHHH',header)
    phoff,shoff=fields[5:7]
    ehsize,phentsize,phnum,shentsize,shnum,_=fields[8:]
    require(ehsize==64,'MALFORMED_ELF_HEADER_SIZE')
    if shoff and (shnum==0 or phnum==0xffff):
        require(shentsize==64,'MALFORMED_EXTENDED_ELF_SECTION_SIZE')
        first=struct.unpack(order+'IIQQQQIIQQ',bounded_read(stream,shoff,64,extent))
        if shnum==0:shnum=first[5]
        if phnum==0xffff:phnum=first[7]
    else:
        require(phnum!=0xffff,'MISSING_ELF_EXTENDED_INDEX_SECTION')
    require(0<=phnum<=2048 and 0<=shnum<=4096,'ELF_TABLE_LIMIT')
    require((phnum==0 or (phoff>=64 and phentsize==56)) and
            ((not shoff and shnum==0) or (shoff>=64 and shnum>0 and shentsize==64)),
            'MALFORMED_ELF_NOTE_TABLE_LAYOUT')
    ranges=set()
    if phnum:
        for phdr in struct.iter_unpack(order+'IIQQQQQQ',bounded_read(stream,phoff,56*phnum,extent)):
            if phdr[0]==4 and phdr[5]:
                ranges.add((phdr[2],phdr[5]))
    if shnum:
        for shdr in struct.iter_unpack(order+'IIQQQQIIQQ',bounded_read(stream,shoff,64*shnum,extent)):
            if shdr[1]==7 and shdr[5]:
                ranges.add((shdr[4],shdr[5]))
    require(len(ranges)<=128 and sum(size for _,size in ranges)<=4<<20,'BOUNDED_ELF_NOTE_RANGES')
    found={}
    for offset,size in sorted(ranges):
        for position,digest in scan_range(bounded_read(stream,offset,size,extent),offset,order).items():
            require(found.setdefault(position,digest)==digest,'CONFLICTING_ELF_NOTE_ALIASES')
    return dict(ranged_note_regions=len(ranges),GNU_physical_build_note_count=len(found),
                GNU_physical_build_ids=list(found.values()))


def note_result(stats,rc,stdout,stderr):
    raw=stdout
    marker=re.compile(rb'\bBuild[ \t]+ID[ \t]*:',re.IGNORECASE)
    found=list(marker.finditer(raw))
    lines=[re.match(rb'[ \t]*([0-9a-fA-F]{8,128})(?=[ \t\r\n]|$)',raw[item.end():]) for item in found]
    valid=[item[1].decode().lower() for item in lines if item and len(item[1])%2==0]
    result=dict(stats,readelf_notes_rc=rc,readelf_stdout_bytes=len(raw),readelf_stderr_bytes=len(stderr),
                readelf_stdout_SHA256=sha256(raw).hexdigest(),readelf_stderr_SHA256=sha256(stderr).hexdigest(),
                readelf_text_markers=len(found),readelf_text_valid_IDs=len(valid),
                readelf_GNU_tag=bool(re.search(rb'\bNT_GNU_BUILD_ID\b',raw)),build_id=None,qualified=False)
    raw_ids=stats['GNU_physical_build_ids']
    if rc or stderr.strip():
        result['status']='READELF_FAILED_OR_DIAGNOSTIC'
    elif len(raw_ids)>1 or len(found)>1:
        result['status']='AMBIGUOUS_DUPLICATE_GNU_BUILD_ID'
    elif len(found)!=len(valid):
        result['status']='MALFORMED_GNU_BUILD_ID_TEXT'
    elif (not raw_ids and not valid and not result['readelf_GNU_tag']
          and not re.search(rb'\bBuild[ \t]+ID\b',raw,re.IGNORECASE)):
        result.update(status='ABSENT_CONFIRMED',qualified=True)
    elif len(raw_ids)==len(valid)==1 and raw_ids==valid:
        result.update(status='PRESENT_CONFIRMED',qualified=True,build_id=raw_ids[0])
    else:
        result['status']='READELF_TEXT_BINARY_PARSER_MISMATCH'
    return result


def probe(path,pinned,trace,readelf):
    """Call only after pinning exact resolved real path and full file SHA/extent."""
    rc,stdout,stderr=trace([readelf,'-nW','--',str(path)],with_stderr=True)
    def identity(a):return (a.st_dev,a.st_ino,a.st_size,a.st_mtime_ns,a.st_ctime_ns)
    known=tuple(pinned['file_identity'])
    try:
        require(identity(path.lstat())==known and pinned['bytes']==known[2],'LIBRARY_CHANGED_BEFORE_NOTE_READ')
        with path.open('rb') as stream:
            require(identity(os.fstat(stream.fileno()))==known,'LIBRARY_CHANGED_AT_NOTE_OPEN')
            data=file_notes(stream,known[2])
            require(identity(os.fstat(stream.fileno()))==known,'LIBRARY_CHANGED_DURING_NOTE_READ')
        require(identity(path.lstat())==known,'LIBRARY_CHANGED_AFTER_NOTE_READ')
    except NoteError as error:
        return dict(status=str(error),qualified=False,build_id=None,readelf_notes_rc=rc,readelf_stdout_bytes=len(stdout),
                    readelf_stderr_bytes=len(stderr),readelf_stdout_SHA256=sha256(stdout).hexdigest(),
                    readelf_stderr_SHA256=sha256(stderr).hexdigest())
    return note_result(data,rc,stdout,stderr)
