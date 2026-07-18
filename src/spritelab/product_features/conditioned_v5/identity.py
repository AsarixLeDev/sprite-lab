"""Exact production-code inventory for conditioned Dataset-v5 trust bindings."""

from __future__ import annotations

import ast
import base64
import csv
import hashlib
import importlib.metadata
import io
import os
import re
import stat
import sys
import unicodedata
import zlib
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any, Final

from spritelab.training.campaign import stable_hash
from spritelab.utils.pinned_executable import PinnedExecutableError, read_executable_identity
from spritelab.utils.safe_fs import AnchoredDirectory, UnsafeFilesystemOperation

CODE_INVENTORY_SCHEMA = "spritelab.dataset.conditioned-code-inventory.v3"
AUDITOR_INVENTORY_SCHEMA = "spritelab.dataset.conditioned-auditor-inventory.v3"
WORKER_RUNTIME_SCHEMA = "spritelab.dataset.conditioned-worker-runtime.v1"

# ``-I`` ignores every Python startup variable, ``-S`` prevents site and .pth
# execution, and ``-B`` prevents bytecode writes.  The bootstrap retains the
# interpreter's fixed standard-library paths and inserts only this checkout's
# audited ``src`` root before executing the exact audited worker file.
WORKER_BOOTSTRAP_SOURCE: Final = r"""
import hashlib,importlib.machinery,importlib.util,io,json,os,stat,sys,types

def _same(left,right):
    return (left.st_dev,left.st_ino,stat.S_IFMT(left.st_mode),left.st_size,left.st_mtime_ns,left.st_nlink)==(right.st_dev,right.st_ino,stat.S_IFMT(right.st_mode),right.st_size,right.st_mtime_ns,right.st_nlink) and left.st_nlink==1

def _bound_bytes(path,expected_sha256,expected_size):
    expected_size=int(expected_size)
    before=os.lstat(path)
    reparse=getattr(before,"st_file_attributes",0)&getattr(stat,"FILE_ATTRIBUTE_REPARSE_POINT",0x400)
    if not stat.S_ISREG(before.st_mode) or stat.S_ISLNK(before.st_mode) or reparse or before.st_nlink!=1 or before.st_size!=expected_size:
        raise RuntimeError("audited child code is unsafe")
    descriptor=os.open(path,os.O_RDONLY|getattr(os,"O_BINARY",0)|getattr(os,"O_NOFOLLOW",0))
    try:
        opened=os.fstat(descriptor)
        if not _same(before,opened):
            raise RuntimeError("audited child code changed while opening")
        chunks=[]
        remaining=expected_size+1
        while remaining:
            chunk=os.read(descriptor,min(1024*1024,remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining-=len(chunk)
        after_open=os.fstat(descriptor)
    finally:
        os.close(descriptor)
    payload=b"".join(chunks)
    after=os.lstat(path)
    if len(payload)!=expected_size or hashlib.sha256(payload).hexdigest()!=expected_sha256 or not _same(before,after_open) or not _same(before,after):
        raise RuntimeError("audited child code changed while reading")
    return payload

def _unique_object(pairs):
    value={}
    for key,item in pairs:
        if key in value:
            raise RuntimeError("audited module manifest repeats a key")
        value[key]=item
    return value

def _manifest_path(source_root,relative):
    if not isinstance(relative,str) or not relative or "\\" in relative or relative.startswith("/"):
        raise RuntimeError("audited module path is invalid")
    parts=relative.split("/")
    if any(part in ("",".","..") for part in parts) or parts[0]!="spritelab":
        raise RuntimeError("audited module path is invalid")
    value=os.path.abspath(os.path.join(source_root,*parts))
    if os.path.commonpath((source_root,value))!=source_root:
        raise RuntimeError("audited module path escapes source root")
    return value

def _runtime_manifest_path(runtime_root,relative):
    if not isinstance(relative,str) or not relative or "\\" in relative or relative.startswith("/") or ":" in relative or "\x00" in relative:
        raise RuntimeError("audited dependency path is invalid")
    parts=relative.split("/")
    leading=0
    saw_name=False
    for part in parts:
        if part=="..":
            if saw_name:
                raise RuntimeError("audited dependency path has embedded traversal")
            leading+=1
        elif part in ("", "."):
            raise RuntimeError("audited dependency path is invalid")
        else:
            saw_name=True
    if not saw_name or leading>4:
        raise RuntimeError("audited dependency path escapes its bounded root")
    allowed=runtime_root
    for _index in range(leading):
        allowed=os.path.dirname(allowed)
    value=os.path.abspath(os.path.join(runtime_root,*parts))
    if os.path.commonpath((allowed,value))!=allowed:
        raise RuntimeError("audited dependency path escapes its bounded root")
    return value

def _stable_hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=True).encode("utf-8")).hexdigest()

source_root,manifest_path,manifest_sha256,manifest_size,runtime_root_count,*remaining=sys.argv[1:]
source_root=os.path.abspath(source_root)
source_metadata=os.lstat(source_root)
source_reparse=getattr(source_metadata,"st_file_attributes",0)&getattr(stat,"FILE_ATTRIBUTE_REPARSE_POINT",0x400)
if not os.path.isabs(source_root) or not stat.S_ISDIR(source_metadata.st_mode) or stat.S_ISLNK(source_metadata.st_mode) or source_reparse:
    raise RuntimeError("audited source root is unsafe")
manifest_bytes=_bound_bytes(manifest_path,manifest_sha256,manifest_size)
try:
    runtime_root_count=int(runtime_root_count)
except ValueError as error:
    raise RuntimeError("audited runtime root count is invalid") from error
if runtime_root_count<1 or len(remaining)<runtime_root_count*3+1:
    raise RuntimeError("audited runtime roots are unavailable")
runtime_roots=[]
for index in range(runtime_root_count):
    path,device,inode=remaining[index*3:index*3+3]
    path=os.path.abspath(path)
    metadata=os.lstat(path)
    reparse=getattr(metadata,"st_file_attributes",0)&getattr(stat,"FILE_ATTRIBUTE_REPARSE_POINT",0x400)
    if not os.path.isabs(path) or not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode) or reparse or metadata.st_dev!=int(device) or metadata.st_ino!=int(inode):
        raise RuntimeError("audited runtime root identity changed")
    runtime_roots.append((path,int(device),int(inode)))
worker_args=remaining[runtime_root_count*3:]
if len(worker_args)!=4:
    raise RuntimeError("audited worker arguments are invalid")
workspace=os.path.abspath(worker_args[1])
if workspace!=os.path.abspath(os.getcwd()):
    raise RuntimeError("audited worker workspace differs from its process root")
pycache_parent=os.path.join(workspace,"tmp")
pycache_root=os.path.join(pycache_parent,"dependency-pycache")
parent_metadata=os.lstat(pycache_parent)
parent_reparse=getattr(parent_metadata,"st_file_attributes",0)&getattr(stat,"FILE_ATTRIBUTE_REPARSE_POINT",0x400)
if not stat.S_ISDIR(parent_metadata.st_mode) or stat.S_ISLNK(parent_metadata.st_mode) or parent_reparse:
    raise RuntimeError("audited dependency pycache parent is unsafe")
os.mkdir(pycache_root,0o700)
pycache_metadata=os.lstat(pycache_root)
if not stat.S_ISDIR(pycache_metadata.st_mode) or stat.S_ISLNK(pycache_metadata.st_mode) or os.listdir(pycache_root):
    raise RuntimeError("audited dependency pycache root is not verified empty")
sys.pycache_prefix=pycache_root
sys.dont_write_bytecode=True
try:
    manifest=json.loads(manifest_bytes.decode("utf-8"),object_pairs_hook=_unique_object)
except Exception as error:
    raise RuntimeError("audited module manifest is invalid") from error
if not isinstance(manifest,dict) or set(manifest)!={"schema_version","worker_module","helper_module","modules","resource_packages","runtime_dependencies"} or manifest.get("schema_version")!="spritelab.dataset.conditioned-worker-module-manifest.v2":
    raise RuntimeError("audited module manifest schema is invalid")
modules=manifest.get("modules")
if not isinstance(modules,dict) or not modules:
    raise RuntimeError("audited module manifest is empty")
bound={}
for fullname,entry in modules.items():
    if not isinstance(fullname,str) or not (fullname=="spritelab" or fullname.startswith("spritelab.")):
        raise RuntimeError("audited module name is invalid")
    if not isinstance(entry,dict) or set(entry)!={"relative_path","sha256","byte_count","is_package"}:
        raise RuntimeError("audited module binding is invalid")
    digest=entry.get("sha256")
    size=entry.get("byte_count")
    is_package=entry.get("is_package")
    if not isinstance(digest,str) or len(digest)!=64 or any(ch not in "0123456789abcdef" for ch in digest) or not isinstance(size,int) or isinstance(size,bool) or size<=0 or not isinstance(is_package,bool):
        raise RuntimeError("audited module binding is invalid")
    bound[fullname]=(_manifest_path(source_root,entry.get("relative_path")),digest,size,is_package)
resource_packages=manifest.get("resource_packages")
if not isinstance(resource_packages,dict):
    raise RuntimeError("audited resource package manifest is invalid")
bound_resources={}
for package,entries in resource_packages.items():
    if package not in bound or not bound[package][3] or not isinstance(entries,dict):
        raise RuntimeError("audited resource package binding is invalid")
    resources={}
    for name,entry in entries.items():
        if not isinstance(name,str) or not name or "/" in name or "\\" in name or name in (".",".."):
            raise RuntimeError("audited resource name is invalid")
        if not isinstance(entry,dict) or set(entry)!={"relative_path","sha256","byte_count"}:
            raise RuntimeError("audited resource binding is invalid")
        digest=entry.get("sha256")
        size=entry.get("byte_count")
        if not isinstance(digest,str) or len(digest)!=64 or any(ch not in "0123456789abcdef" for ch in digest) or not isinstance(size,int) or isinstance(size,bool) or size<=0:
            raise RuntimeError("audited resource binding is invalid")
        resources[name]=(_manifest_path(source_root,entry.get("relative_path")),digest,size)
    bound_resources[package]=resources
runtime_dependencies=manifest.get("runtime_dependencies")
if not isinstance(runtime_dependencies,dict) or not runtime_dependencies:
    raise RuntimeError("audited dependency inventories are unavailable")
bound_dependencies={}
bound_dependency_directories=set()
for distribution,binding in runtime_dependencies.items():
    if not isinstance(distribution,str) or not distribution or not isinstance(binding,dict) or set(binding)!={"runtime_root_index","inventory"}:
        raise RuntimeError("audited dependency binding is invalid")
    root_index=binding.get("runtime_root_index")
    inventory=binding.get("inventory")
    if not isinstance(root_index,int) or isinstance(root_index,bool) or root_index<0 or root_index>=len(runtime_roots) or not isinstance(inventory,dict):
        raise RuntimeError("audited dependency root binding is invalid")
    inventory_base=dict(inventory)
    inventory_identity=inventory_base.pop("inventory_sha256",None)
    expected_inventory_keys={"schema_version","distribution","version","record_relative_path","record_sha256","record_declared_paths","record_file_count","owned_roots","files","file_count","unrecorded_file_count","total_bytes","paths_exposed"}
    if set(inventory_base)!=expected_inventory_keys or inventory.get("schema_version")!="spritelab.runtime.installed-distribution-inventory.v2" or inventory.get("paths_exposed") is not False or not isinstance(inventory_identity,str) or _stable_hash(inventory_base)!=inventory_identity:
        raise RuntimeError("audited dependency inventory identity is invalid")
    files=inventory.get("files")
    declared_paths=inventory.get("record_declared_paths")
    owned_roots=inventory.get("owned_roots")
    if not isinstance(files,dict) or not files or not isinstance(declared_paths,list) or len(declared_paths)!=len(set(declared_paths)) or inventory.get("record_file_count")!=len(declared_paths) or not set(declared_paths)<=set(files) or not isinstance(owned_roots,list) or not owned_roots or inventory.get("file_count")!=len(files) or inventory.get("unrecorded_file_count")!=len(files)-len(declared_paths) or inventory.get("total_bytes")!=sum(entry.get("byte_count",-1) for entry in files.values() if isinstance(entry,dict)):
        raise RuntimeError("audited dependency file inventory is invalid")
    for root_entry in owned_roots:
        if not isinstance(root_entry,dict) or set(root_entry)!={"relative_path","kind"} or root_entry.get("kind") not in ("directory","file") or not isinstance(root_entry.get("relative_path"),str):
            raise RuntimeError("audited dependency owned-root inventory is invalid")
    runtime_root=runtime_roots[root_index][0]
    for relative,entry in files.items():
        if not isinstance(entry,dict) or set(entry)!={"sha256","byte_count"}:
            raise RuntimeError("audited dependency file binding is invalid")
        digest=entry.get("sha256")
        size=entry.get("byte_count")
        if not isinstance(digest,str) or len(digest)!=64 or any(ch not in "0123456789abcdef" for ch in digest) or not isinstance(size,int) or isinstance(size,bool) or size<0:
            raise RuntimeError("audited dependency file binding is invalid")
        dependency_path=_runtime_manifest_path(runtime_root,relative)
        _bound_bytes(dependency_path,digest,size)
        try:
            inside=os.path.commonpath((runtime_root,dependency_path))==runtime_root
        except ValueError:
            inside=False
        if not inside:
            continue
        key=os.path.normcase(dependency_path)
        existing=bound_dependencies.get(key)
        value=(dependency_path,digest,size)
        if existing is not None and existing!=value:
            raise RuntimeError("audited dependency file collision")
        bound_dependencies[key]=value
        directory=os.path.dirname(dependency_path)
        while os.path.commonpath((runtime_root,directory))==runtime_root:
            bound_dependency_directories.add(os.path.normcase(directory))
            if directory==runtime_root:
                break
            directory=os.path.dirname(directory)
if not bound_dependencies:
    raise RuntimeError("audited dependency file allowlist is empty")
worker_module=manifest.get("worker_module")
helper_module=manifest.get("helper_module")
if worker_module not in bound or helper_module not in bound:
    raise RuntimeError("audited worker bindings are unavailable")

class _BoundLoader:
    def __init__(self,fullname,binding):
        self.fullname=fullname
        self.binding=binding
    def create_module(self,spec):
        return None
    def exec_module(self,module):
        path,digest,size,_is_package=self.binding
        payload=_bound_bytes(path,digest,size)
        module.__file__=path
        exec(compile(payload,path,"exec",dont_inherit=True),module.__dict__)
    def get_filename(self,fullname):
        if fullname!=self.fullname:
            raise ImportError("audited loader module mismatch")
        return self.binding[0]
    def is_package(self,fullname):
        if fullname!=self.fullname:
            raise ImportError("audited loader module mismatch")
        return self.binding[3]
    def get_resource_reader(self,fullname):
        if fullname!=self.fullname or fullname not in bound_resources:
            return None
        return _BoundResourceReader(fullname,bound_resources[fullname])

class _BoundResourceReader:
    def __init__(self,fullname,resources):
        self.fullname=fullname
        self.resources=resources
    def open_resource(self,name):
        entry=self.resources.get(name)
        if entry is None:
            raise FileNotFoundError(name)
        return io.BytesIO(_bound_bytes(*entry))
    def resource_path(self,name):
        raise FileNotFoundError("audited resources are pathless")
    def is_resource(self,name):
        return name in self.resources
    def contents(self):
        return tuple(sorted(self.resources))

def _dependency_binding(path):
    value=os.path.abspath(os.fspath(path))
    binding=bound_dependencies.get(os.path.normcase(value))
    if binding is None:
        raise ImportError("unbound runtime dependency file refused")
    return binding

def _inside_runtime_root(path):
    value=os.path.abspath(os.fspath(path))
    for root,_device,_inode in runtime_roots:
        try:
            if os.path.commonpath((root,value))==root:
                return True
        except ValueError:
            pass
    return False

def _open_pinned_dependency(path,digest,size):
    before=os.lstat(path)
    reparse=getattr(before,"st_file_attributes",0)&getattr(stat,"FILE_ATTRIBUTE_REPARSE_POINT",0x400)
    if not stat.S_ISREG(before.st_mode) or stat.S_ISLNK(before.st_mode) or reparse or before.st_nlink!=1 or before.st_size!=size:
        raise ImportError("audited native dependency is unsafe")
    if os.name=="nt":
        import ctypes,msvcrt
        kernel32=ctypes.WinDLL("kernel32",use_last_error=True)
        create_file=kernel32.CreateFileW
        create_file.argtypes=(ctypes.c_wchar_p,ctypes.c_uint32,ctypes.c_uint32,ctypes.c_void_p,ctypes.c_uint32,ctypes.c_uint32,ctypes.c_void_p)
        create_file.restype=ctypes.c_void_p
        close_handle=kernel32.CloseHandle
        close_handle.argtypes=(ctypes.c_void_p,)
        close_handle.restype=ctypes.c_int
        handle=create_file(path,0x80000000,0x1,None,3,0x80,None)
        if handle in (None,ctypes.c_void_p(-1).value):
            raise ImportError("audited native dependency could not be pinned")
        try:
            descriptor=msvcrt.open_osfhandle(int(handle),os.O_RDONLY|getattr(os,"O_BINARY",0))
        except Exception:
            close_handle(handle)
            raise
    else:
        descriptor=os.open(path,os.O_RDONLY|getattr(os,"O_NOFOLLOW",0))
        try:
            import fcntl
            fcntl.flock(descriptor,fcntl.LOCK_SH|fcntl.LOCK_NB)
        except Exception:
            os.close(descriptor)
            raise ImportError("audited native dependency could not be read-locked")
    try:
        opened=os.fstat(descriptor)
        if not _same(before,opened):
            raise ImportError("audited native dependency changed while pinning")
        calculated=hashlib.sha256()
        byte_count=0
        while byte_count<=size:
            chunk=os.read(descriptor,min(1024*1024,size+1-byte_count))
            if not chunk:
                break
            calculated.update(chunk)
            byte_count+=len(chunk)
        os.lseek(descriptor,0,os.SEEK_SET)
        after_open=os.fstat(descriptor)
        after=os.lstat(path)
        if byte_count!=size or calculated.hexdigest()!=digest or not _same(before,after_open) or not _same(before,after):
            raise ImportError("audited native dependency changed while pinning")
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise

class _DependencySourceLoader:
    def __init__(self,fullname,binding,is_package):
        self.fullname=fullname
        self.binding=binding
        self.package=is_package
    def create_module(self,spec):
        return None
    def exec_module(self,module):
        path,digest,size=self.binding
        payload=_bound_bytes(path,digest,size)
        module.__file__=path
        module.__cached__=None
        exec(compile(payload,path,"exec",dont_inherit=True),module.__dict__)
    def get_filename(self,fullname):
        if fullname!=self.fullname:
            raise ImportError("audited dependency loader module mismatch")
        return self.binding[0]
    def is_package(self,fullname):
        if fullname!=self.fullname:
            raise ImportError("audited dependency loader module mismatch")
        return self.package
    def get_data(self,path):
        return _bound_bytes(*_dependency_binding(path))
    def get_resource_reader(self,fullname):
        if fullname!=self.fullname or not self.package:
            return None
        return _DependencyResourceReader(os.path.dirname(self.binding[0]))

class _DependencyResourceReader:
    def __init__(self,directory):
        self.directory=os.path.abspath(directory)
    def _entry(self,name):
        if not isinstance(name,str) or not name or "/" in name or "\\" in name or name in (".",".."):
            raise FileNotFoundError(name)
        path=os.path.abspath(os.path.join(self.directory,name))
        if os.path.dirname(path)!=self.directory:
            raise FileNotFoundError(name)
        return _dependency_binding(path)
    def open_resource(self,name):
        return io.BytesIO(_bound_bytes(*self._entry(name)))
    def resource_path(self,name):
        raise FileNotFoundError("audited dependency resources are pathless")
    def is_resource(self,name):
        try:
            self._entry(name)
        except (FileNotFoundError,ImportError):
            return False
        return True
    def contents(self):
        values=[]
        for path,_digest,_size in bound_dependencies.values():
            if os.path.dirname(path)==self.directory and not path.lower().endswith((".pyc",".pyo")):
                values.append(os.path.basename(path))
        return tuple(sorted(set(values)))

class _PinnedExtensionLoader:
    def __init__(self,fullname,binding):
        self.fullname=fullname
        self.binding=binding
        self.descriptor=None
        self.delegate=None
    def create_module(self,spec):
        path,digest,size=self.binding
        self.descriptor=_open_pinned_dependency(path,digest,size)
        self.delegate=importlib.machinery.ExtensionFileLoader(self.fullname,path)
        delegate_spec=importlib.util.spec_from_file_location(self.fullname,path,loader=self.delegate)
        if delegate_spec is None:
            os.close(self.descriptor)
            self.descriptor=None
            raise ImportError("audited native dependency spec is unavailable")
        try:
            return self.delegate.create_module(delegate_spec)
        except BaseException:
            os.close(self.descriptor)
            self.descriptor=None
            raise
    def exec_module(self,module):
        if self.delegate is None or self.descriptor is None:
            raise ImportError("audited native dependency was not pinned")
        try:
            self.delegate.exec_module(module)
            _bound_bytes(*self.binding)
        finally:
            os.close(self.descriptor)
            self.descriptor=None

class _DependencyFinder:
    def find_spec(self,fullname,path=None,target=None):
        if fullname=="spritelab" or fullname.startswith("spritelab."):
            return None
        spec=importlib.machinery.PathFinder.find_spec(fullname,path,target)
        if spec is None:
            return None
        locations=tuple(spec.submodule_search_locations or ())
        runtime_locations=[value for value in locations if _inside_runtime_root(value)]
        if spec.origin in (None,"built-in","frozen"):
            if runtime_locations and any(os.path.normcase(os.path.abspath(value)) not in bound_dependency_directories for value in runtime_locations):
                raise ImportError("unbound runtime dependency namespace refused")
            return spec
        origin=os.path.abspath(os.fspath(spec.origin))
        if not _inside_runtime_root(origin):
            return spec
        if origin.lower().endswith((".pyc",".pyo")):
            raise ImportError("runtime dependency bytecode execution refused")
        binding=_dependency_binding(origin)
        if origin.lower().endswith(".py"):
            loader=_DependencySourceLoader(fullname,binding,bool(locations))
            return importlib.util.spec_from_loader(fullname,loader,is_package=bool(locations))
        if any(origin.lower().endswith(suffix.lower()) for suffix in importlib.machinery.EXTENSION_SUFFIXES):
            loader=_PinnedExtensionLoader(fullname,binding)
            return importlib.util.spec_from_loader(fullname,loader,is_package=False)
        raise ImportError("unsupported runtime dependency module origin refused")

class _BoundFinder:
    def find_spec(self,fullname,path=None,target=None):
        if fullname=="spritelab" or fullname.startswith("spritelab."):
            binding=bound.get(fullname)
            if binding is None:
                raise ImportError("unbound spritelab import refused")
            loader=_BoundLoader(fullname,binding)
            return importlib.util.spec_from_loader(fullname,loader,is_package=binding[3])
        return None

if any(name=="spritelab" or name.startswith("spritelab.") for name in sys.modules):
    raise RuntimeError("spritelab imported before audited finder")
sys.meta_path.insert(0,_DependencyFinder())
sys.meta_path.insert(0,_BoundFinder())
sys._spritelab_conditioned_runtime_roots=tuple(runtime_roots)
worker_path,worker_sha256,worker_size,_worker_package=bound[worker_module]
helper_path,helper_sha256,helper_size,_helper_package=bound[helper_module]
worker_bytes=_bound_bytes(worker_path,worker_sha256,worker_size)
helper_bytes=_bound_bytes(helper_path,helper_sha256,helper_size)
helper_name="_spritelab_conditioned_write_confinement"
helper=types.ModuleType(helper_name)
helper.__file__=helper_path
helper.__package__=""
sys.modules[helper_name]=helper
exec(compile(helper_bytes,helper.__file__,"exec",dont_inherit=True),helper.__dict__)
main=types.ModuleType("__main__")
main.__file__=worker_path
main.__package__=""
sys.modules["__main__"]=main
sys.argv=[worker_path,*worker_args]
exec(compile(worker_bytes,main.__file__,"exec",dont_inherit=True),main.__dict__)
""".lstrip()


def _compressed_worker_bootstrap_command(source: str) -> str:
    """Encode the exact audited bootstrap below the Windows command-line limit."""

    compressed_hex = zlib.compress(source.encode("utf-8"), level=9).hex()
    return (
        "import zlib;"
        "exec(compile(zlib.decompress(bytes.fromhex("
        f"{compressed_hex!r}"
        ")).decode('utf-8'),'<spritelab-conditioned-bootstrap>','exec',dont_inherit=True),"
        "globals(),globals())"
    )


WORKER_BOOTSTRAP_COMMAND_SOURCE: Final = _compressed_worker_bootstrap_command(WORKER_BOOTSTRAP_SOURCE)
WORKER_INTERPRETER_FLAGS: Final = ("-I", "-S", "-B", "-c")
WORKER_INHERITED_ENVIRONMENT_KEYS: Final = ("SystemRoot", "WINDIR")
_WORKER_ENVIRONMENT_POLICY: Final = {
    "schema_version": "spritelab.dataset.conditioned-worker-environment-policy.v1",
    "interpreter_flags": list(WORKER_INTERPRETER_FLAGS),
    "bootstrap_sha256": hashlib.sha256(WORKER_BOOTSTRAP_SOURCE.encode("utf-8")).hexdigest(),
    "bootstrap_transport": "zlib-level-9-hex-v1",
    "bootstrap_command_sha256": hashlib.sha256(WORKER_BOOTSTRAP_COMMAND_SOURCE.encode("utf-8")).hexdigest(),
    "inherited_environment_keys": list(WORKER_INHERITED_ENVIRONMENT_KEYS),
    "fixed_environment_keys": ["TEMP", "TMP", "TMPDIR"],
    "environment_default": "absent",
    "stderr": "null-device",
    "cwd": "conditioned-private-workspace",
    "runtime_import_roots": "descriptor-rehashed-exact-distribution-manifests-after-write-confinement",
    "first_party_import_policy": "exact-inventory-only-descriptor-bound",
    "third_party_import_policy": "exact-inventory-source-resource-loader-and-pinned-native-loader",
    "bytecode_policy": "verified-empty-private-pycache-prefix-and-runtime-pyc-refusal",
}


def controlled_worker_launch_arguments() -> tuple[str, ...]:
    """Return the exact launch arguments only while they match audited policy."""

    flags = tuple(WORKER_INTERPRETER_FLAGS)
    source = WORKER_BOOTSTRAP_SOURCE
    command_source = WORKER_BOOTSTRAP_COMMAND_SOURCE
    policy = _WORKER_ENVIRONMENT_POLICY
    if (
        list(flags) != policy.get("interpreter_flags")
        or hashlib.sha256(source.encode("utf-8")).hexdigest() != policy.get("bootstrap_sha256")
        or policy.get("bootstrap_transport") != "zlib-level-9-hex-v1"
        or hashlib.sha256(command_source.encode("utf-8")).hexdigest() != policy.get("bootstrap_command_sha256")
    ):
        raise ConditionedCodeIdentityError("The controlled worker launch source differs from its audited policy.")
    return (*flags, command_source)


_AUDITOR_MODULES = {
    "label_audit": (
        "product_features/conditioned_v5/audit_runner.py",
        "dataset_v5/audits.py",
        "dataset_v5/blind.py",
        "dataset_v5/conservative_labeling.py",
        "dataset_v5/evidence.py",
    ),
    "dataset_validation": (
        "product_features/conditioned_v5/audit_runner.py",
        "codec/validate.py",
        "dataset_maker/qa.py",
        "dataset_maker/training_manifest_qa.py",
        "dataset_v5/named_views.py",
    ),
}

TRUSTED_AUDITOR_IDS = {
    "label_audit": "spritelab.dataset-v5-audits",
    "dataset_validation": "spritelab.dataset-v5-validation",
}

_PRODUCTION_ENTRYPOINTS: Final = (
    "product_features/conditioned_v5/intake.py",
    "product_features/conditioned_v5/service.py",
)

_PRODUCTION_INTEGRATION_MODULES: Final = (
    "config/__init__.py",
    "product_runtime.py",
    "product_features/conditioned_v5/__init__.py",
    "product_features/conditioned_v5/legacy_worker.py",
    "product_features/conditioned_v5/plugin.py",
    "product_features/conditioned_v5/web.py",
)

_PRODUCTION_RESOURCE_PATHS: Final = (
    "config/hallucination_denylist.yaml",
    "config/sheet_mappings.yaml",
    "config/source_profiles.yaml",
    "config/taxonomy.yaml",
)

_RUNTIME_DISTRIBUTIONS: Final = (
    "anyio",
    "idna",
    "numpy",
    "Pillow",
    "PyYAML",
    "setuptools",
    "starlette",
    "typing_extensions",
)
CALLBACK_RUNTIME_SCHEMA: Final = "spritelab.dataset.conditioned-callback-runtime.v1"
_DISTRIBUTION_INVENTORY_SCHEMA: Final = "spritelab.runtime.installed-distribution-inventory.v2"
_MAX_DISTRIBUTION_FILES: Final = 20_000
_MAX_DISTRIBUTION_FILE_BYTES: Final = 512 * 1024 * 1024
_MAX_DISTRIBUTION_TOTAL_BYTES: Final = 2 * 1024 * 1024 * 1024
_MAX_DISTRIBUTION_PARENT_ESCAPES: Final = 4


class ConditionedCodeIdentityError(ValueError):
    """The conditioned production implementation cannot be inventoried safely."""


def conditioned_code_inventory() -> dict[str, Any]:
    """Hash every production module that enforces import, build, and publication."""

    package_root = Path(__file__).resolve(strict=True).parents[2]
    files: dict[str, dict[str, Any]] = {}
    for relative in conditioned_code_module_paths():
        path = package_root.joinpath(*relative.split("/"))
        content = _read_single_link(path)
        files[f"spritelab/{relative}"] = {
            "sha256": hashlib.sha256(content).hexdigest(),
            "byte_count": len(content),
        }
    for relative in _PRODUCTION_RESOURCE_PATHS:
        path = package_root.joinpath(*relative.split("/"))
        content = _read_single_link(path)
        files[f"spritelab/{relative}"] = {
            "sha256": hashlib.sha256(content).hexdigest(),
            "byte_count": len(content),
        }
    runtime_dependencies = _runtime_dependency_inventories()
    payload = {
        "schema_version": CODE_INVENTORY_SCHEMA,
        "files": dict(sorted(files.items())),
        "file_count": len(files),
        "total_bytes": sum(int(item["byte_count"]) for item in files.values()),
        "runtime_dependencies": runtime_dependencies,
        "worker_runtime": controlled_worker_runtime(runtime_dependencies=runtime_dependencies),
    }
    return {**payload, "inventory_sha256": stable_hash(payload)}


def conditioned_code_identity() -> str:
    return str(conditioned_code_inventory()["inventory_sha256"])


def conditioned_callback_runtime_inventory(
    code_inventory: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Bind the callback to the exact dependency and controlled-worker runtime."""

    inventory = conditioned_code_inventory() if code_inventory is None else code_inventory
    runtime_dependencies = inventory.get("runtime_dependencies")
    worker_runtime = inventory.get("worker_runtime")
    if not isinstance(runtime_dependencies, Mapping) or not isinstance(worker_runtime, Mapping):
        raise ConditionedCodeIdentityError("The conditioned callback runtime inventory is incomplete.")
    payload = {
        "schema_version": CALLBACK_RUNTIME_SCHEMA,
        "runtime_dependencies": dict(runtime_dependencies),
        "worker_runtime": dict(worker_runtime),
    }
    return {**payload, "runtime_identity_sha256": stable_hash(payload)}


def controlled_worker_executable() -> Path:
    """Return the exact interpreter file selected for the isolated child."""

    try:
        executable = Path(sys.executable).resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ConditionedCodeIdentityError("The controlled worker interpreter is unavailable.") from exc
    if not executable.is_file():
        raise ConditionedCodeIdentityError("The controlled worker interpreter is not a regular file.")
    return executable


def controlled_worker_environment(temporary: Path) -> dict[str, str]:
    """Build the child's minimal environment without inheriting Python/provider state."""

    temporary_value = os.fspath(temporary)
    if not temporary_value or not Path(temporary_value).is_absolute():
        raise ConditionedCodeIdentityError("The controlled worker temporary directory must be absolute.")
    environment = {"TEMP": temporary_value, "TMP": temporary_value, "TMPDIR": temporary_value}
    if os.name == "nt":
        for name in WORKER_INHERITED_ENVIRONMENT_KEYS:
            value = os.environ.get(name)
            if value:
                environment[name] = value
    return environment


def controlled_worker_runtime(*, runtime_dependencies: Mapping[str, Mapping[str, Any]] | None = None) -> dict[str, Any]:
    """Return path-free evidence for the interpreter and fixed launch policy."""

    executable = controlled_worker_executable()
    try:
        executable_identity = read_executable_identity(executable)
    except PinnedExecutableError as exc:
        raise ConditionedCodeIdentityError("The controlled worker interpreter is unsafe.") from exc
    policy = dict(_WORKER_ENVIRONMENT_POLICY)
    policy_identity = stable_hash(policy)
    roots = controlled_worker_dependency_roots()
    dependencies = _runtime_dependency_inventories() if runtime_dependencies is None else dict(runtime_dependencies)
    dependency_inventory_identities = {
        name: str(inventory.get("inventory_sha256") or "") for name, inventory in sorted(dependencies.items())
    }
    if set(dependency_inventory_identities) != set(_RUNTIME_DISTRIBUTIONS) or any(
        not re.fullmatch(r"[0-9a-f]{64}", value) for value in dependency_inventory_identities.values()
    ):
        raise ConditionedCodeIdentityError("The controlled worker dependency inventory is incomplete.")
    root_evidence = [
        {
            "device": metadata.st_dev,
            "inode": metadata.st_ino,
            "distributions": list(distributions),
        }
        for _path, metadata, distributions in roots
    ]
    runtime = {
        "schema_version": WORKER_RUNTIME_SCHEMA,
        "implementation": sys.implementation.name,
        "cache_tag": sys.implementation.cache_tag,
        "version_hex": sys.hexversion,
        "executable_sha256": executable_identity.executable_sha256,
        "executable_byte_count": executable_identity.byte_count,
        "executable_metadata_sha256": executable_identity.metadata_sha256,
        "environment_policy": policy,
        "environment_policy_identity": policy_identity,
        "dependency_roots": root_evidence,
        "dependency_roots_identity": stable_hash(root_evidence),
        "runtime_dependency_inventory_identities": dependency_inventory_identities,
        "runtime_dependencies_identity": stable_hash(dependencies),
        "paths_exposed": False,
    }
    return {**runtime, "runtime_identity": stable_hash(runtime)}


def controlled_worker_dependency_roots() -> tuple[tuple[Path, os.stat_result, tuple[str, ...]], ...]:
    """Return safe distribution roots plus pathless identities for child launch."""

    by_path: dict[Path, list[str]] = {}
    for distribution_name in _RUNTIME_DISTRIBUTIONS:
        try:
            distribution = importlib.metadata.distribution(distribution_name)
            root = Path(distribution.locate_file("")).resolve(strict=True)
        except (importlib.metadata.PackageNotFoundError, OSError, RuntimeError) as exc:
            raise ConditionedCodeIdentityError(
                f"The conditioned runtime dependency {distribution_name!r} is unavailable."
            ) from exc
        by_path.setdefault(root, []).append(distribution_name)
    roots: list[tuple[Path, os.stat_result, tuple[str, ...]]] = []
    for root, distributions in sorted(by_path.items(), key=lambda item: os.fspath(item[0])):
        metadata = root.lstat()
        reparse = getattr(metadata, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
        if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode) or reparse:
            raise ConditionedCodeIdentityError("A conditioned runtime dependency root is unsafe.")
        roots.append((root, metadata, tuple(sorted(distributions))))
    return tuple(roots)


def conditioned_code_module_paths() -> tuple[str, ...]:
    """Resolve the first-party import closure used by conditioned production code."""

    return _module_closure((*_PRODUCTION_ENTRYPOINTS, *_PRODUCTION_INTEGRATION_MODULES))


def _module_closure(entrypoints: tuple[str, ...]) -> tuple[str, ...]:
    package_root = Path(__file__).resolve(strict=True).parents[2]
    discovered: set[str] = set()
    pending = list(entrypoints)
    while pending:
        relative = pending.pop()
        if relative in discovered:
            continue
        payload = _read_single_link(package_root.joinpath(*relative.split("/")))
        discovered.add(relative)
        imports = set(_first_party_imports(relative, payload, package_root))
        imports.update(_parent_initializers(relative, package_root))
        for imported in imports:
            if imported not in discovered:
                pending.append(imported)
    return tuple(sorted(discovered))


def trusted_auditor_inventory(kind: str) -> dict[str, Any]:
    """Return the exact repository implementation inventory trusted for one report kind."""

    relative_paths = _AUDITOR_MODULES.get(kind)
    auditor_id = TRUSTED_AUDITOR_IDS.get(kind)
    if relative_paths is None or auditor_id is None:
        raise ConditionedCodeIdentityError("The conditioned auditor kind is unsupported.")
    package_root = Path(__file__).resolve(strict=True).parents[2]
    files: dict[str, dict[str, Any]] = {}
    for relative in _module_closure(relative_paths):
        payload = _read_single_link(package_root.joinpath(*relative.split("/")))
        files[f"spritelab/{relative}"] = {
            "sha256": hashlib.sha256(payload).hexdigest(),
            "byte_count": len(payload),
        }
    base = {
        "schema_version": AUDITOR_INVENTORY_SCHEMA,
        "auditor_id": auditor_id,
        "files": dict(sorted(files.items())),
        "file_count": len(files),
        "total_bytes": sum(int(item["byte_count"]) for item in files.values()),
        "runtime_dependencies": _runtime_dependency_inventories(),
        "interpreter_runtime": _auditor_interpreter_runtime(),
    }
    return {**base, "inventory_sha256": stable_hash(base)}


def _auditor_interpreter_runtime() -> dict[str, Any]:
    try:
        identity = read_executable_identity(sys.executable)
    except PinnedExecutableError as exc:
        raise ConditionedCodeIdentityError("The trusted auditor interpreter is unsafe.") from exc
    payload = {
        "schema_version": "spritelab.dataset.conditioned-auditor-runtime.v1",
        "implementation": sys.implementation.name,
        "cache_tag": sys.implementation.cache_tag,
        "version_hex": sys.hexversion,
        "executable_sha256": identity.executable_sha256,
        "executable_byte_count": identity.byte_count,
        "executable_metadata_sha256": identity.metadata_sha256,
        "paths_exposed": False,
    }
    return {**payload, "runtime_identity": stable_hash(payload)}


def _read_single_link(path: Path) -> bytes:
    try:
        before = path.lstat()
    except OSError as exc:
        raise ConditionedCodeIdentityError("A conditioned production module is unavailable.") from exc
    reparse = getattr(before, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    if not stat.S_ISREG(before.st_mode) or stat.S_ISLNK(before.st_mode) or reparse or before.st_nlink != 1:
        raise ConditionedCodeIdentityError("Conditioned production modules must be single-link regular files.")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        if not _same_file(before, opened):
            raise ConditionedCodeIdentityError("A conditioned production module changed while opening.")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            payload = handle.read(before.st_size + 1)
        opened_after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    after = path.lstat()
    if len(payload) != before.st_size or not _same_file(before, opened_after) or not _same_file(before, after):
        raise ConditionedCodeIdentityError("A conditioned production module changed while hashing.")
    return payload


def _hash_regular_file(path: Path, *, label: str) -> tuple[str, int]:
    """Hash one exact no-follow file while proving its inode stays stable."""

    try:
        before = path.lstat()
    except OSError as exc:
        raise ConditionedCodeIdentityError(f"The {label} is unavailable.") from exc
    reparse = getattr(before, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    if not stat.S_ISREG(before.st_mode) or stat.S_ISLNK(before.st_mode) or reparse:
        raise ConditionedCodeIdentityError(f"The {label} must be a regular non-linked file.")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ConditionedCodeIdentityError(f"The {label} could not be opened safely.") from exc
    digest = hashlib.sha256()
    byte_count = 0
    try:
        opened = os.fstat(descriptor)
        if not _same_file(before, opened):
            raise ConditionedCodeIdentityError(f"The {label} changed while opening.")
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            byte_count += len(chunk)
        opened_after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    try:
        after = path.lstat()
    except OSError as exc:
        raise ConditionedCodeIdentityError(f"The {label} changed while hashing.") from exc
    if byte_count != before.st_size or not _same_file(before, opened_after) or not _same_file(before, after):
        raise ConditionedCodeIdentityError(f"The {label} changed while hashing.")
    return digest.hexdigest(), byte_count


def _first_party_imports(relative: str, payload: bytes, package_root: Path) -> tuple[str, ...]:
    try:
        tree = ast.parse(payload.decode("utf-8"), filename=relative)
    except (SyntaxError, UnicodeDecodeError) as exc:
        raise ConditionedCodeIdentityError("A conditioned production module could not be parsed.") from exc
    imports: set[str] = set()
    current = PureModulePath(relative)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported = _resolve_absolute_module(alias.name, package_root)
                if imported is not None:
                    imports.add(imported)
        elif isinstance(node, ast.ImportFrom):
            module_name = node.module or ""
            if node.level:
                module_name = current.resolve_relative(module_name, node.level)
            imported = _resolve_absolute_module(module_name, package_root)
            if imported is not None:
                imports.add(imported)
            if node.level and not node.module:
                for alias in node.names:
                    child = _resolve_absolute_module(f"{module_name}.{alias.name}", package_root)
                    if child is not None:
                        imports.add(child)
        elif isinstance(node, ast.Call):
            literal = _literal_dynamic_import(node)
            if literal is not None:
                imported = _resolve_absolute_module(literal, package_root)
                if imported is not None:
                    imports.add(imported)
    return tuple(sorted(imports))


class PureModulePath:
    """Small import-name resolver that never imports production modules."""

    def __init__(self, relative: str) -> None:
        path = Path(relative)
        parts = list(path.with_suffix("").parts)
        if parts[-1] == "__init__":
            parts.pop()
        else:
            parts.pop()
        self.package_parts = tuple(parts)

    def resolve_relative(self, module: str, level: int) -> str:
        keep = len(self.package_parts) - (level - 1)
        if keep < 0:
            raise ConditionedCodeIdentityError("A conditioned production module has an invalid relative import.")
        parts = ("spritelab", *self.package_parts[:keep])
        if module:
            parts = (*parts, *module.split("."))
        return ".".join(parts)


def _resolve_absolute_module(module_name: str, package_root: Path) -> str | None:
    if module_name == "spritelab":
        return "__init__.py" if (package_root / "__init__.py").is_file() else None
    if not module_name.startswith("spritelab."):
        return None
    relative = module_name.removeprefix("spritelab.").replace(".", "/")
    module = package_root / f"{relative}.py"
    if module.is_file():
        return f"{relative}.py"
    package = package_root / relative / "__init__.py"
    if package.is_file():
        return f"{relative}/__init__.py"
    return None


def _parent_initializers(relative: str, package_root: Path) -> tuple[str, ...]:
    path = Path(relative)
    directory = path.parent
    parents: list[str] = []
    while directory.parts:
        initializer = directory / "__init__.py"
        if (package_root / initializer).is_file():
            parents.append(initializer.as_posix())
        directory = directory.parent
    root_initializer = package_root / "__init__.py"
    if root_initializer.is_file():
        parents.append("__init__.py")
    return tuple(parents)


def _literal_dynamic_import(node: ast.Call) -> str | None:
    if not node.args or not isinstance(node.args[0], ast.Constant) or not isinstance(node.args[0].value, str):
        return None
    function = node.func
    if isinstance(function, ast.Name) and function.id == "__import__":
        return node.args[0].value
    if isinstance(function, ast.Name) and function.id == "import_module":
        return node.args[0].value
    if (
        isinstance(function, ast.Attribute)
        and function.attr == "import_module"
        and isinstance(function.value, ast.Name)
        and function.value.id == "importlib"
    ):
        return node.args[0].value
    return None


def _runtime_dependency_inventories() -> dict[str, dict[str, Any]]:
    return {name: _installed_distribution_inventory(name) for name in sorted(_RUNTIME_DISTRIBUTIONS, key=str.casefold)}


def installed_distribution_inventory(distribution_name: str) -> dict[str, Any]:
    """Return path-free, descriptor-rehashed evidence for one installed wheel."""

    if not isinstance(distribution_name, str) or not distribution_name.strip():
        raise ConditionedCodeIdentityError("An installed distribution name is required.")
    return _installed_distribution_inventory(distribution_name.strip())


def _installed_distribution_inventory(distribution_name: str) -> dict[str, Any]:
    """Bind RECORD plus every file below its distribution-owned top-level roots."""

    try:
        distribution = importlib.metadata.distribution(distribution_name)
        raw_root = Path(distribution.locate_file(""))
        root = raw_root.resolve(strict=True)
        raw_metadata_root = Path(distribution._path)
        metadata_root = raw_metadata_root.resolve(strict=True)
    except (AttributeError, importlib.metadata.PackageNotFoundError, OSError, RuntimeError, TypeError) as exc:
        raise ConditionedCodeIdentityError(
            f"The conditioned runtime dependency {distribution_name!r} is unavailable."
        ) from exc
    if Path(os.path.abspath(raw_root)) != root or Path(os.path.abspath(raw_metadata_root)) != metadata_root:
        raise ConditionedCodeIdentityError("A conditioned runtime dependency crosses a link or reparse seam.")
    record_path = metadata_root / "RECORD"
    try:
        record_relative = Path(os.path.relpath(record_path, root)).as_posix()
    except ValueError as exc:
        raise ConditionedCodeIdentityError("A runtime dependency RECORD is outside its installation volume.") from exc
    record_bytes = _read_single_link(record_path)
    declarations = _parse_distribution_record(record_bytes)
    if record_relative not in declarations:
        raise ConditionedCodeIdentityError("A runtime dependency RECORD does not bind itself.")
    if not 0 < len(declarations) <= _MAX_DISTRIBUTION_FILES:
        raise ConditionedCodeIdentityError("A runtime dependency RECORD exceeds its file-count bound.")

    declared_bindings: dict[str, tuple[str | None, int | None]] = {}
    owned_roots: dict[str, tuple[Path, str, bool]] = {}
    for record_name, declaration in sorted(declarations.items()):
        record_relative_path, parent_escapes = _canonical_distribution_record_path(record_name)
        path = _located_distribution_record_file(root, record_relative_path, parent_escapes)
        canonical = _public_distribution_inventory_path(record_relative_path, parent_escapes)
        if canonical in declared_bindings:
            raise ConditionedCodeIdentityError("A runtime dependency RECORD contains a canonical path collision.")
        declared_bindings[canonical] = declaration
        if parent_escapes:
            root_relative = canonical
            owned_path = path
        else:
            root_relative = PurePosixPath(canonical).parts[0]
            owned_path = root / root_relative
        metadata = owned_path.lstat()
        if _metadata_is_link_or_reparse(metadata):
            raise ConditionedCodeIdentityError("A runtime dependency owned root crosses a link or reparse seam.")
        if stat.S_ISDIR(metadata.st_mode):
            kind = "directory"
        elif stat.S_ISREG(metadata.st_mode) and int(metadata.st_nlink) == 1:
            kind = "file"
        else:
            raise ConditionedCodeIdentityError("A runtime dependency owned root is not regular and unlinked.")
        inside_installation = parent_escapes == 0
        existing = owned_roots.get(root_relative)
        if existing is not None and existing != (owned_path, kind, inside_installation):
            raise ConditionedCodeIdentityError("A runtime dependency owned root is ambiguous.")
        owned_roots[root_relative] = (owned_path, kind, inside_installation)

    files: dict[str, dict[str, Any]] = {}
    collision_keys: set[str] = set()
    total_bytes = 0
    owned_root_evidence: list[dict[str, str]] = []
    try:
        with AnchoredDirectory(root, root) as installation_anchor:
            for root_relative, (owned_path, kind, inside_installation) in sorted(owned_roots.items()):
                owned_root_evidence.append({"relative_path": root_relative, "kind": kind})
                scanned = _scan_distribution_owned_root(
                    owned_path,
                    root_relative=root_relative,
                    kind=kind,
                    installation_anchor=installation_anchor if inside_installation else None,
                )
                for relative, binding in scanned.items():
                    collision = unicodedata.normalize("NFC", relative).casefold()
                    if relative in files or collision in collision_keys:
                        raise ConditionedCodeIdentityError(
                            "A runtime dependency owned inventory contains a path collision."
                        )
                    collision_keys.add(collision)
                    files[relative] = binding
                    total_bytes += int(binding["byte_count"])
                    if len(files) > _MAX_DISTRIBUTION_FILES:
                        raise ConditionedCodeIdentityError(
                            "A runtime dependency inventory exceeds its file-count bound."
                        )
                    if total_bytes > _MAX_DISTRIBUTION_TOTAL_BYTES:
                        raise ConditionedCodeIdentityError(
                            "A runtime dependency inventory exceeds its total byte bound."
                        )
            installation_anchor.verify()
    except (OSError, UnsafeFilesystemOperation) as exc:
        raise ConditionedCodeIdentityError("A runtime dependency installation root cannot be anchored safely.") from exc

    for canonical, declaration in sorted(declared_bindings.items()):
        binding = files.get(canonical)
        if binding is None:
            raise ConditionedCodeIdentityError("A runtime dependency RECORD file is outside its owned roots.")
        declared_hash, declared_size = declaration
        byte_count = int(binding["byte_count"])
        digest = str(binding["sha256"])
        if declared_size is not None and declared_size != byte_count:
            raise ConditionedCodeIdentityError("A runtime dependency file differs from its RECORD size.")
        if declared_hash is not None:
            encoded = base64.urlsafe_b64encode(bytes.fromhex(digest)).decode("ascii").rstrip("=")
            if encoded != declared_hash:
                raise ConditionedCodeIdentityError("A runtime dependency file differs from its RECORD digest.")
    if _read_single_link(record_path) != record_bytes:
        raise ConditionedCodeIdentityError("A runtime dependency RECORD changed while it was inventoried.")
    version = str(distribution.version or "")
    canonical_name = str(distribution.metadata.get("Name") or distribution_name)
    if not version or not canonical_name:
        raise ConditionedCodeIdentityError("A runtime dependency lacks canonical package metadata.")
    base = {
        "schema_version": _DISTRIBUTION_INVENTORY_SCHEMA,
        "distribution": canonical_name,
        "version": version,
        "record_relative_path": record_relative,
        "record_sha256": hashlib.sha256(record_bytes).hexdigest(),
        "record_declared_paths": sorted(declared_bindings),
        "record_file_count": len(declared_bindings),
        "owned_roots": owned_root_evidence,
        "files": dict(sorted(files.items())),
        "file_count": len(files),
        "unrecorded_file_count": len(files) - len(declared_bindings),
        "total_bytes": total_bytes,
        "paths_exposed": False,
    }
    return {**base, "inventory_sha256": stable_hash(base)}


def _scan_distribution_owned_root(
    path: Path,
    *,
    root_relative: str,
    kind: str,
    installation_anchor: AnchoredDirectory | None,
) -> dict[str, dict[str, Any]]:
    if kind == "file":
        if installation_anchor is not None:
            if "/" in root_relative or root_relative in {"", ".", ".."}:
                raise ConditionedCodeIdentityError("A runtime dependency owned file root is invalid.")
            metadata = installation_anchor.lstat(root_relative)
            digest, byte_count = _hash_anchored_distribution_file(
                installation_anchor,
                root_relative,
                expected=metadata,
            )
        else:
            try:
                with AnchoredDirectory(path.parent, path.parent) as parent_anchor:
                    metadata = parent_anchor.lstat(path.name)
                    digest, byte_count = _hash_anchored_distribution_file(
                        parent_anchor,
                        path.name,
                        expected=metadata,
                    )
            except (OSError, UnsafeFilesystemOperation) as exc:
                raise ConditionedCodeIdentityError(
                    "A runtime dependency external owned file cannot be anchored safely."
                ) from exc
        if byte_count > _MAX_DISTRIBUTION_FILE_BYTES:
            raise ConditionedCodeIdentityError("A runtime dependency file exceeds its byte bound.")
        return {root_relative: {"sha256": digest, "byte_count": byte_count}}
    if kind != "directory":
        raise ConditionedCodeIdentityError("A runtime dependency owned-root kind is invalid.")
    files: dict[str, dict[str, Any]] = {}
    try:
        if installation_anchor is None or "/" in root_relative or root_relative in {"", ".", ".."}:
            raise ConditionedCodeIdentityError("A runtime dependency directory root is not installation-anchored.")
        with installation_anchor.open_directory_immovable(root_relative) as root_anchor:
            _scan_distribution_owned_anchor(
                root_anchor,
                relative_directory=root_relative,
                expected_device=int(root_anchor.directory_metadata().st_dev),
                files=files,
                depth=0,
            )
    except (OSError, UnsafeFilesystemOperation) as exc:
        raise ConditionedCodeIdentityError("A runtime dependency owned directory cannot be anchored safely.") from exc
    return dict(sorted(files.items()))


def _scan_distribution_owned_anchor(
    anchor: AnchoredDirectory,
    *,
    relative_directory: str,
    expected_device: int,
    files: dict[str, dict[str, Any]],
    depth: int,
) -> None:
    if depth > 64:
        raise ConditionedCodeIdentityError("A runtime dependency owned directory is too deeply nested.")
    anchor.verify()
    if int(anchor.directory_metadata().st_dev) != expected_device:
        raise ConditionedCodeIdentityError("A runtime dependency owned root crosses a filesystem device.")
    names = anchor.names()
    collisions: set[str] = set()
    for name in names:
        collision = unicodedata.normalize("NFC", name).casefold()
        if collision in collisions:
            raise ConditionedCodeIdentityError("A runtime dependency directory contains a name collision.")
        collisions.add(collision)
        metadata = anchor.lstat(name)
        if _metadata_is_link_or_reparse(metadata) or int(metadata.st_dev) != expected_device:
            raise ConditionedCodeIdentityError("A runtime dependency owned root crosses a filesystem seam.")
        relative = f"{relative_directory}/{name}"
        if stat.S_ISDIR(metadata.st_mode):
            with anchor.open_directory_immovable(name) as child:
                _scan_distribution_owned_anchor(
                    child,
                    relative_directory=relative,
                    expected_device=expected_device,
                    files=files,
                    depth=depth + 1,
                )
            continue
        if not stat.S_ISREG(metadata.st_mode) or int(metadata.st_nlink) != 1:
            raise ConditionedCodeIdentityError("A runtime dependency owned root contains a non-owned entry.")
        digest, byte_count = _hash_anchored_distribution_file(anchor, name, expected=metadata)
        if byte_count > _MAX_DISTRIBUTION_FILE_BYTES:
            raise ConditionedCodeIdentityError("A runtime dependency file exceeds its byte bound.")
        files[relative] = {"sha256": digest, "byte_count": byte_count}
        if len(files) > _MAX_DISTRIBUTION_FILES:
            raise ConditionedCodeIdentityError("A runtime dependency inventory exceeds its file-count bound.")
    anchor.verify()


def _hash_anchored_distribution_file(
    anchor: AnchoredDirectory,
    name: str,
    *,
    expected: os.stat_result,
) -> tuple[str, int]:
    descriptor = anchor.open_file(name, os.O_RDONLY | int(getattr(os, "O_BINARY", 0)))
    digest = hashlib.sha256()
    byte_count = 0
    try:
        opened = os.fstat(descriptor)
        if not _same_file(expected, opened):
            raise ConditionedCodeIdentityError("A runtime dependency file changed while opening.")
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            byte_count += len(chunk)
        opened_after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if (
        byte_count != int(expected.st_size)
        or not _same_file(expected, opened_after)
        or not _same_file(expected, anchor.lstat(name))
    ):
        raise ConditionedCodeIdentityError("A runtime dependency file changed while hashing.")
    return digest.hexdigest(), byte_count


def _metadata_is_link_or_reparse(metadata: os.stat_result) -> bool:
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return stat.S_ISLNK(metadata.st_mode) or bool(int(getattr(metadata, "st_file_attributes", 0)) & reparse_flag)


def _parse_distribution_record(payload: bytes) -> dict[str, tuple[str | None, int | None]]:
    try:
        text = payload.decode("utf-8")
        rows = list(csv.reader(io.StringIO(text, newline="")))
    except (UnicodeDecodeError, csv.Error) as exc:
        raise ConditionedCodeIdentityError("A runtime dependency RECORD is invalid.") from exc
    declarations: dict[str, tuple[str | None, int | None]] = {}
    for row in rows:
        if len(row) != 3:
            raise ConditionedCodeIdentityError("A runtime dependency RECORD row is malformed.")
        name, raw_hash, raw_size = row
        if not name or name in declarations:
            raise ConditionedCodeIdentityError("A runtime dependency RECORD contains duplicate entries.")
        digest: str | None = None
        if raw_hash:
            algorithm, separator, value = raw_hash.partition("=")
            if algorithm != "sha256" or separator != "=" or not re.fullmatch(r"[A-Za-z0-9_-]{43}", value):
                raise ConditionedCodeIdentityError("A runtime dependency RECORD digest is unsupported.")
            digest = value
        size: int | None = None
        if raw_size:
            if not raw_size.isdecimal():
                raise ConditionedCodeIdentityError("A runtime dependency RECORD size is invalid.")
            size = int(raw_size)
            if size > _MAX_DISTRIBUTION_FILE_BYTES:
                raise ConditionedCodeIdentityError("A runtime dependency RECORD size exceeds its byte bound.")
        declarations[name] = (digest, size)
    return declarations


def _canonical_distribution_record_path(value: str) -> tuple[str, int]:
    if not value or "\\" in value or "\x00" in value or ":" in value:
        raise ConditionedCodeIdentityError("A runtime dependency RECORD path is unsafe.")
    path = PurePosixPath(value)
    if path.is_absolute() or path.as_posix() != value:
        raise ConditionedCodeIdentityError("A runtime dependency RECORD path is not canonical POSIX.")
    parent_escapes = 0
    saw_name = False
    for part in path.parts:
        if part == "..":
            if saw_name:
                raise ConditionedCodeIdentityError("A runtime dependency RECORD path has embedded traversal.")
            parent_escapes += 1
        else:
            saw_name = True
    if not saw_name or parent_escapes > _MAX_DISTRIBUTION_PARENT_ESCAPES:
        raise ConditionedCodeIdentityError("A runtime dependency RECORD path escapes its bounded installation root.")
    return value, parent_escapes


def _public_distribution_inventory_path(value: str, parent_escapes: int) -> str:
    """Encode bounded RECORD parent entries without publishing traversal paths."""

    if parent_escapes == 0:
        return value
    parts = PurePosixPath(value).parts[parent_escapes:]
    if not parts:
        raise ConditionedCodeIdentityError("A runtime dependency RECORD path has no public file name.")
    return PurePosixPath("external", f"parent-{parent_escapes}", *parts).as_posix()


def _located_distribution_record_file(root: Path, relative: str, parent_escapes: int) -> Path:
    lexical = Path(os.path.abspath(root.joinpath(*PurePosixPath(relative).parts)))
    allowed_root = root
    for _ in range(parent_escapes):
        allowed_root = allowed_root.parent
    try:
        lexical.relative_to(allowed_root)
        resolved = lexical.resolve(strict=True)
    except (OSError, RuntimeError, ValueError) as exc:
        raise ConditionedCodeIdentityError("A runtime dependency RECORD entry is unavailable or escapes.") from exc
    if resolved != lexical:
        raise ConditionedCodeIdentityError("A runtime dependency RECORD entry crosses a link or reparse seam.")
    return lexical


def _distribution_version(distribution: str) -> str:
    try:
        version = importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError as exc:
        raise ConditionedCodeIdentityError(
            f"The conditioned runtime dependency {distribution!r} is unavailable."
        ) from exc
    if not version:
        raise ConditionedCodeIdentityError("A conditioned runtime dependency version is unavailable.")
    return version


def _same_file(left: os.stat_result, right: os.stat_result) -> bool:
    return (
        left.st_dev == right.st_dev
        and left.st_ino == right.st_ino
        and left.st_size == right.st_size
        and left.st_mtime_ns == right.st_mtime_ns
        and left.st_nlink == right.st_nlink == 1
    )


__all__ = [
    "AUDITOR_INVENTORY_SCHEMA",
    "CODE_INVENTORY_SCHEMA",
    "TRUSTED_AUDITOR_IDS",
    "ConditionedCodeIdentityError",
    "conditioned_code_identity",
    "conditioned_code_inventory",
    "conditioned_code_module_paths",
    "controlled_worker_dependency_roots",
    "installed_distribution_inventory",
    "trusted_auditor_inventory",
]
