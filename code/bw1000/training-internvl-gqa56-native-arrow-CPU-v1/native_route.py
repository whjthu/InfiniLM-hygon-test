"""Call the deployed Swift SFT argument/dataset path and DataLoaderMixin without a model."""
from copy import copy
import importlib
import inspect
from pathlib import Path
from types import SimpleNamespace

from lineage import ObservedOccurrences, arrow_projection, collate_observed, need, verify_rank


SWIFT=Path("/swift-runtime/site/swift")
SOURCES={
    "dataset/loader.py":(18212,"bfd75ce24bf0b2ba2fe414a37afd1ce2465575269689325791481eb2bef6d9b4"),
    "pipelines/train/sft.py":(15931,"024ff2ff4e6755a07c30bcc3dd081106816b31fbb30169d9e27125f0b67be7d0"),
}


def rewrite_args(command,exposure,processor,output):
    need(len(command)>2 and command[1]=="sft","EXACT_ORIGINAL_SFT_ENTRY")
    argv=command[2:].copy()
    for key,value in (("--dataset",str(exposure)),("--model",str(processor)),("--output_dir",str(output))):
        need(argv.count(key)==1 and argv.index(key)+1<len(argv),"EXACT_ORIGINAL_SFT_CLI_BIND")
        argv[argv.index(key)+1]=value
    # These are the original timing-only observer and its registration, not the dataset arguments.
    for key,expected in (("--external_plugins","/home/whj/nvidia-baseline/swift_callback_nvidia_baseline_v2.py"),
                         ("--callbacks","nvidia_baseline")):
        need(argv.count(key)==1,"EXACT_ORIGINAL_OBSERVER_CLI")
        offset=argv.index(key)
        need(argv[offset+1:offset+2]==[expected] and
             (offset+2==len(argv) or argv[offset+2].startswith("--")),"EXACT_ORIGINAL_OBSERVER_ONLY_CLI")
        del argv[offset:offset+2]
    return argv


def actual_swift(pin):
    for rel,(size,wanted) in SOURCES.items():
        pin(SWIFT/rel,size,wanted)
    sft=importlib.import_module("swift.pipelines.train.sft")
    dataset=importlib.import_module("swift.dataset.loader")
    need(Path(sft.__file__)==SWIFT/"pipelines/train/sft.py" and
         Path(dataset.__file__)==SWIFT/"dataset/loader.py","EXACT_NATIVE_SFT_AND_DATASET_ORIGINS")
    cls=sft.SwiftSft
    get_data=cls._get_dataset
    need(inspect.getsourcefile(get_data)==str(SWIFT/"pipelines/train/sft.py") and
         list(inspect.signature(get_data).parameters)==["self"] and callable(dataset.load_dataset),
         "EXACT_NATIVE_SFT_DATASET_METHOD")
    return cls,get_data


def original_loader(command,exposure,processor,output,cls,get_data):
    holder=object.__new__(cls)
    parse=holder._parse_args
    need(len(inspect.signature(parse).parameters)==1,"EXACT_NATIVE_SFT_PARSER_SHAPE")
    arg=parse(rewrite_args(command,exposure,processor,output))
    need(tuple(arg.dataset)==(str(exposure),) and not arg.val_dataset and arg.split_dataset_ratio==0
         and arg.dataset_shuffle is arg.train_dataloader_shuffle is True and arg.seed==arg.data_seed==42
         and arg.per_device_train_batch_size==1 and arg.gradient_accumulation_steps==4 and arg.max_steps==25
         and arg.dataloader_drop_last is arg.packing is arg.padding_free is arg.group_by_length is False
         and arg.dataloader_num_workers==4 and not arg.deepspeed,"ORIGINAL_TRAIN_ONLY_SWIFT_OPTS_REQUIRED")
    def load(shuffle):
        current=copy(arg)
        current.dataset_shuffle=shuffle
        holder.args=current
        output=get_data(holder)
        need(isinstance(output,tuple) and len(output)==2 and output[0] is not None and output[1] is None,
             "ORIGINAL_NATIVE_DATASET_TRAIN_ONLY")
        return output[0]
    return arg,load


def observe_rank(torch, mixin, impl, dataset_type, parsed, load, rank, expected, reference):
    dist=torch.distributed
    need(dist.is_initialized() and (dist.get_rank(),dist.get_world_size())==(rank,8) and
         not torch.cuda.is_initialized(),"EXACT_NATIVE_CPU_RANK_CONTEXT")
    unshuffled,shuffled=load(False),load(True)
    order,messages,values=arrow_projection(unshuffled,shuffled,dataset_type,expected)
    # The native loader receives the real shuffled Arrow-backed dataset via a metadata-only projection.
    metadata=ObservedOccurrences(messages,shuffled._data.table,order)
    traversal=copy(parsed)
    traversal.dataloader_num_workers=0
    traversal.dataloader_persistent_workers=False
    traversal.dataloader_prefetch_factor=None
    traversal.dataloader_pin_memory=False
    holder=SimpleNamespace(template=SimpleNamespace(sequence_parallel_size=1),train_dataset=metadata,
                           _train_batch_size=1,data_collator=collate_observed,accelerator=SimpleNamespace(device=torch.device("cpu")),
                           args=traversal)
    need(method_globals_are_actual(mixin.get_train_dataloader,impl),"EXACT_REAL_SHARD_GLOBALS")
    loader=mixin.get_train_dataloader(holder,skip_batches=0)
    need(type(loader) is impl.DataLoaderShard and type(loader.batch_sampler) is impl.BatchSamplerShard
         and loader.dataset is metadata and len(loader)==8 and len(loader.batch_sampler)==8,
         "EXACT_REAL_DATALOADER_SHARD_OBJECT")
    need(loader.batch_sampler.rank==rank and loader.batch_sampler.world_size==8 and
         loader.batch_sampler.total_samples==8,"ACTUAL_REAL_SHARD_RANK_FIELDS")
    observed=[]
    iterator=iter(loader)
    try:
        for item in iterator:
            need(len(observed)<8,"EXTRA_ACTUAL_LOADER_SLOT")
            observed.append(item)
    finally:
        close=getattr(iterator,"close",None)
        if close is not None:
            close()
    need(not torch.cuda.is_initialized(),"NATIVE_ARROW_PROBE_MUST_REMAIN_CPU")
    return verify_rank(rank,observed,reference,order,expected,values),order


def method_globals_are_actual(method,impl):
    return all(method.__globals__.get(name) is getattr(impl,name) for name in ("DataLoaderShard","BatchSamplerShard"))
