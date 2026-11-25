"""Describe both actual providers, then compare source and Arrow in physical order."""

from hashlib import sha256
import dataclasses
import inspect
import json
from pathlib import Path
import re


ROWS=800
COLS=("messages","images")
META_CAP=1048576


def require(ok,code):
    if not ok:
        raise ValueError(code)


def digest(data):
    return sha256(data).hexdigest()


def encoded(value):
    return (json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False)+"\n").encode()


def bounded(data):
    require(len(data)<=META_CAP,"BOUNDED_PROVIDER_METADATA")
    return {"bytes":len(data),"SHA256":digest(data)}


def type_id(value):
    kind=type(value)
    return digest(encoded((kind.__module__,kind.__qualname__)))


def provider(value):
    path=Path(inspect.getfile(value)).resolve(strict=True)
    with path.open("rb") as stream:
        raw=stream.read(META_CAP+1)
    return bounded(raw)


def shapes(value,depth=0):
    category={dict:"DICT",list:"LIST",tuple:"TUPLE",str:"TEXT",bytes:"BYTES",int:"INTEGER",bool:"BOOLEAN",type(None):"NULL"}
    result={"type":category.get(type(value),"OTHER"),"type_SHA256":type_id(value)}
    if type(value) in (str,bytes,list,tuple,dict):
        result["length"]=len(value)
    if depth<2 and type(value) in (list,tuple,dict):
        if type(value) is dict:
            value=tuple(value.values())
        result["leading_shapes"]=[shapes(x,depth+1) for x in value[:4]]
    return result


def declared_shape(datatype,types,depth=0):
    family="OTHER"
    children=[]
    if list_like(datatype,types):
        family="LIST"
        children=[datatype.value_type]
    elif types.is_struct(datatype):
        family="STRUCT"
        children=[field.type for field in datatype]
    elif text_like(datatype,types):
        family="TEXT"
    elif types.is_binary(datatype) or types.is_large_binary(datatype):
        family="BINARY"
    result={"family":family,"type_SHA256":type_id(datatype),"child_count":len(children)}
    if depth<3 and children:
        result["leading_children"]=[declared_shape(child,types,depth+1) for child in children[:6]]
    return result


def loaded_snapshot(dataset,get_formatter,PythonFormatter,pa):
    result={"state":"AVAILABLE","physical_rows":None,"logical_rows":None}
    stage="DATASET_EXTENT"
    try:
        table=dataset._data.table
        result.update(logical_rows=len(dataset),physical_rows=len(table),dataset_type_SHA256=type_id(dataset),
                      table_type_SHA256=type_id(table),data_wrapper_type_SHA256=type_id(dataset._data))
        stage="FEATURE_SCHEMA"
        features=dataset.features
        result.update(features=bounded(encoded(features.to_dict())),feature_type_SHA256=type_id(features),
                      feature_schema=bounded(features.arrow_schema.serialize().to_pybytes()))
        stage="ARROW_SCHEMA"
        result["arrow_schema"]=bounded(table.schema.serialize().to_pybytes())
        result["schema_field_count"]=len(table.schema)
        result["fixed_columns"]={name:{"type_SHA256":type_id(table.schema.field(name).type),
           "schema_SHA256":bounded(encoded(str(table.schema.field(name).type)))["SHA256"],
           "declared_storage":declared_shape(table.schema.field(name).type,pa.types),
           "feature_type_SHA256":type_id(features[name]),"nullable":table.schema.field(name).nullable} for name in COLS}
        stage="DATASET_FORMATTER"
        config=dataset.format
        actual=get_formatter(config["type"],features=features,**config["format_kwargs"])
        no_feature=PythonFormatter(features=None,lazy=False)
        result["formatters"]={"dataset_selected_type_SHA256":type_id(actual),"dataset_format_description":bounded(encoded(config)),
            "factory_provider":provider(get_formatter),"selected_provider":provider(type(actual)),
            "native_nonmedia_type_SHA256":type_id(no_feature),"native_nonmedia_provider":provider(type(no_feature)),
            "native_formatter_uses_features":no_feature.features is not None}
    except (OSError,ValueError,KeyError,TypeError,AttributeError,RuntimeError,IndexError) as error:
        result.update(state="METADATA_INCOMPLETE",boundary=stage,error_type_SHA256=type_id(error))
    return result


def list_like(datatype,types):
    return types.is_list(datatype) or types.is_large_list(datatype) or types.is_fixed_size_list(datatype)


def text_like(datatype,types):
    return types.is_string(datatype) or types.is_large_string(datatype)


def struct_fields(datatype,types,names):
    return types.is_struct(datatype) and datatype.num_fields==len(names) and {field.name for field in datatype}==set(names)


def source_messages(value):
    require(type(value) is list and len(value)==2 and all(type(x) is dict and set(x)=={"role","content"} for x in value),
            "EXACT_TWO_SOURCE_MESSAGE_RECORDS")
    require([x["role"] for x in value]==["user","assistant"] and all(type(v) is str for x in value for v in x.values()),
            "EXACT_SOURCE_ROLE_AND_TEXT_TYPES")
    return tuple((x["role"],x["content"]) for x in value)


def media(value,layout,prefix):
    require(type(value) is list and len(value)==1,"EXACTLY_ONE_UNMODIFIED_MEDIA_ELEMENT")
    item=value[0]
    if layout=="NATIVE_IMAGE_PATH_WITH_NULL_STORAGE" or (layout=="SOURCE" and type(item) is dict):
        require(type(item) is dict and set(item)=={"bytes","path"} and item["bytes"] is None,"NO_DROPPED_OR_EMBEDDED_IMAGE_STORAGE")
        item=item["path"]
    require(type(item) is str and item.startswith(prefix) and re.fullmatch(r"[0-9]{3}\.png",item.removeprefix(prefix)),
            "EXACT_ORIGINAL_TRAIN_MEDIA_PATH")
    return item


def exact(a,b):
    if type(a) is not type(b):
        return False
    if type(a) is dict:
        return a.keys()==b.keys() and all(exact(a[k],b[k]) for k in a)
    if type(a) in (list,tuple):
        return len(a)==len(b) and all(exact(x,y) for x,y in zip(a,b,strict=True))
    return a==b


def decode(unit,formatter,evidence=None,columns=COLS):
    scalars={name:unit.column(name)[0] for name in columns}
    if evidence is not None:
        evidence["scalar_classes"]={k:{"type_SHA256":type_id(v),"declared_type_SHA256":type_id(v.type),"valid":v.is_valid} for k,v in scalars.items()}
    scalar={name:value.as_py() for name,value in scalars.items()}
    if evidence is not None:
        evidence["native_scalar_as_py"]={k:shapes(v) for k,v in scalar.items()}
    table,=unit.to_pylist()
    if evidence is not None:
        evidence["native_arrow_table_return"]={k:shapes(v) for k,v in table.items() if k in columns}
    formatted=formatter.format_row(unit)
    if evidence is not None:
        evidence["native_nonmedia_formatter_return"]={k:shapes(v) for k,v in formatted.items() if k in columns}
    require(exact(scalar,table) and exact(table,formatted),"ARROW_NATIVE_DECODERS_DISAGREE")
    return scalar


PROVIDER_PINS={
    "swift_preprocessor":{"bytes":24438,"SHA256":"55f4751391583048bb9c4eca90f95b5d95923c5d958edd0ffe37be223438fa33"},
    "datasets_json":{"bytes":98124,"SHA256":"d52713f02c9e8f94b085a0e7ca29dc06d66d110e2500806a5541a891519049bc"},
    "datasets_formatter":{"bytes":26626,"SHA256":"f3332fedd7d3cb2103bf3363283cd3ec46f6fdf9f69fa3a49df1a660aff4f8c3"},
}


def no_duplicate_pairs(pairs):
    value={}
    for k,v in pairs:
        require(k not in value,"JSON_DUPLICATE_OBJECT_KEY")
        value[k]=v
    return value


def reject_constant(_):
    raise ValueError("JSON_NONFINITE_CONSTANT")


def strict_json(raw):
    return json.loads(raw,object_pairs_hook=no_duplicate_pairs,parse_constant=reject_constant)


def pin_runtime(datasets_module,formatter_module,swift_core,report):
    row_preprocessor=swift_core.RowPreprocessor
    Json=datasets_module.Json
    formatter=formatter_module.PythonFormatter
    decode=formatter_module.PythonFeaturesDecoder.decode_row
    pins={"swift_preprocessor":provider(row_preprocessor),"datasets_json":provider(Json.decode_example),
          "datasets_formatter":provider(formatter.format_row)}
    report["actual_json_runtime_providers"]=pins
    require(pins==PROVIDER_PINS and provider(decode)==PROVIDER_PINS["datasets_formatter"],
            "R36_EXACT_THREE_DEPLOYED_PROVIDER_BYTES")
    path=Path(inspect.getfile(row_preprocessor)).resolve(strict=True)
    with path.open("rb") as stream:
        swift=stream.read(PROVIDER_PINS["swift_preprocessor"]["bytes"]+1)
    require(bounded(swift)==PROVIDER_PINS["swift_preprocessor"] and swift.splitlines()[261].strip()==b"messages_feature = List(Json())",
            "R36_EXACT_SWIFT_LINE262_JSON_LIST")
    require("ujson_loads" in Json.decode_example.__code__.co_names
            and callable(Json.decode_example.__globals__.get("ujson_loads")),"R36_EXACT_NATIVE_JSON_UJSON_BINDING")
    report["runtime_pins_verified"]=True


def exact_bound_method(instance,cls,name):
    method=getattr(instance,name)
    require(getattr(method,"__self__",None) is instance and getattr(method,"__func__",None) is getattr(cls,name),
            "R37_NATIVE_METHOD_IMPLEMENTATION_CHANGED")


def config_dataclass(instance,expected,nested=()):
    require(dataclasses.is_dataclass(instance),"R37_NATIVE_FEATURE_IS_NOT_DATACLASS")
    slots=dataclasses.fields(instance)
    require({slot.name for slot in slots}==set(nested)|set(expected),"R37_INCOMPLETE_NATIVE_DATACLASS_FIELDS")
    result={slot.name:getattr(instance,slot.name) for slot in slots if slot.name not in nested}
    require(exact(result,expected),"R37_NATIVE_FEATURE_CONFIG_DIFFERS_FROM_FROZEN_DEFAULTS")
    return result


def config_feature(feature,List,Json,expected):
    require(type(feature) is List and type(feature.feature) is Json,"R37_EXACT_NATIVE_LIST_JSON_CLASSES")
    leaf=feature.feature
    exact_bound_method(feature,List,"__repr__")
    for name in ("__call__","decode_example","encode_example","cast_storage"):
        exact_bound_method(leaf,Json,name)
    require(type(leaf.pa_type) is type(expected) and leaf.pa_type.equals(expected)
            and type(leaf()) is type(expected) and leaf().equals(expected),"R37_EXACT_NATIVE_JSON_ARROW_ENCODING")
    return {"List":config_dataclass(feature,{"length":-1,"id":None,"_type":"List"},("feature",)),
            "Json":config_dataclass(leaf,{"decode":True,"id":None,"_type":"Json"})}


def bound_provider_methods(Json,List,Features,Formatter,Decoder):
    native_feature_methods=(Json.__call__,Json.decode_example,Json.encode_example,Json.cast_storage,List.__repr__,
                            Features.__init__,Features.decode_example,Features.__init__.__globals__.get("_visit"))
    native_formatter_methods=(Formatter.__init__,Formatter.format_row,Decoder.__init__,Decoder.decode_row)
    require(all(provider(method)==PROVIDER_PINS["datasets_json"] for method in native_feature_methods)
            and all(provider(method)==PROVIDER_PINS["datasets_formatter"] for method in native_formatter_methods),
            "R37_EXACT_ACTIVE_VISITOR_FEATURE_AND_FORMATTER_METHODS")


def native_message_reader(first,second,projected,pa,datasets_module,formatter_module,report):
    feature=first.features["messages"]
    paired=second.features["messages"]
    Json,List,Features=datasets_module.Json,datasets_module.List,datasets_module.Features
    Formatter,Decoder=formatter_module.PythonFormatter,formatter_module.PythonFeaturesDecoder
    typ=projected.schema.field("messages").type
    expected=pa.json_()
    require(pa.types.is_list(typ) and type(typ.value_type) is type(expected) and typ.value_type.equals(expected)
            and typ.equals(first.features.arrow_schema.field("messages").type)
            and typ.equals(second.features.arrow_schema.field("messages").type),"R37_EXACT_STORAGE_AND_DATASET_MESSAGE_TYPES")
    bound_provider_methods(Json,List,Features,Formatter,Decoder)
    config=config_feature(feature,List,Json,expected)
    require(exact(config,config_feature(paired,List,Json,expected)),"R37_LOADER_MESSAGE_FEATURE_CONFIG_DIFFER")
    scoped=Features({"messages":feature})
    require(type(scoped) is Features and set(scoped)=={"messages"},"R37_SCOPED_FEATURES_HAS_OTHER_COLUMNS")
    native=Formatter(features=scoped,lazy=False)
    require(type(native) is Formatter and type(native.features) is Features and set(native.features)=={"messages"},
            "R37_FORMATTER_FEATURES_HAS_OTHER_COLUMNS")
    decoder=native.python_features_decoder
    require(native.features is scoped and type(decoder) is Decoder and decoder.features is native.features
            and set(decoder.features)=={"messages"} and native.token_per_repo_id is decoder.token_per_repo_id is None,
            "R37_NATIVE_FORMATTER_DECODER_COLUMN_BINDING")
    flags=native.features._column_requires_decoding
    require(type(flags) is dict and set(flags)=={"messages"} and flags["messages"] is True,
            "R37_ONLY_MESSAGES_REQUIRES_DECODING")
    exact_bound_method(native,Formatter,"format_row")
    exact_bound_method(decoder,Decoder,"decode_row")
    exact_bound_method(native.features,Features,"decode_example")
    held=native.features["messages"]
    require(exact(config,config_feature(held,List,Json,expected)),"R37_REBUILT_NATIVE_FEATURE_CONFIG_DIFFERS")
    require(native.features.arrow_schema.field("messages").type.equals(typ),"R37_REBUILT_FEATURE_ARROW_ENCODING_DIFFERS")
    require(held is not feature,"R37_EXPECTED_EXACT_NATIVE_LIST_REBUILD_UNOBSERVED")
    require(held.feature is feature.feature,"R37_EXPECTED_EXACT_NATIVE_JSON_LEAF_ORIGIN_UNOBSERVED")
    report["native_formatter_clone"]={"only_messages":True,"native_decoder_shares_features":True,
        "messages_decoding_enabled":True,"outer_list_rebuilt":True,"native_method_implementations_matched":True,
        "native_config_fields_SHA256":digest(encoded(config)),"storage_and_scoped_arrow_types_matched":True,
        "original_json_leaf_retained_by_native_visitor":True}
    return held.feature,native


def verify_json_message_values(unit,stored,source,feature,formatter,evidence=None):
    require(type(stored) is list and len(stored)==2 and all(type(item) is str for item in stored),
            "R36_EXACT_TWO_NATIVE_JSON_STORAGE_STRINGS")
    strict=[strict_json(item) for item in stored]
    if evidence is not None:
        evidence["strict_json_structure"]=shapes(strict)
    # Type and membership are checked before calling the native decoder.
    source_messages(strict)
    source_messages(source)
    direct=[feature.decode_example(item) for item in stored]
    if evidence is not None:
        evidence["native_leaf_decoding_structure"]=shapes(direct)
    from_formatter=formatter.format_row(unit.select(["messages"]))
    if evidence is not None:
        evidence["native_scoped_feature_formatter_structure"]=shapes(from_formatter)
    require(type(from_formatter) is dict and set(from_formatter)=={"messages"}
            and exact(strict,direct) and exact(direct,from_formatter["messages"]) and exact(strict,source),
            "R36_STRICT_NATIVE_AND_ORIGINAL_MESSAGE_VALUES_DIFFER")


ROW_FAILURES=(OSError,ValueError,KeyError,TypeError,AttributeError,RuntimeError,IndexError,OverflowError)
SAFE_CODES={
    "R36_EXACT_THREE_DEPLOYED_PROVIDER_BYTES","R36_EXACT_SWIFT_LINE262_JSON_LIST","R36_EXACT_NATIVE_JSON_UJSON_BINDING",
    "R36_EXACT_TWO_NATIVE_JSON_STORAGE_STRINGS","R36_STRICT_NATIVE_AND_ORIGINAL_MESSAGE_VALUES_DIFFER",
    "R37_EXACT_STORAGE_AND_DATASET_MESSAGE_TYPES","R37_EXACT_NATIVE_LIST_JSON_CLASSES","R37_NATIVE_METHOD_IMPLEMENTATION_CHANGED",
    "R37_SCOPED_FEATURES_HAS_OTHER_COLUMNS","R37_FORMATTER_FEATURES_HAS_OTHER_COLUMNS","R37_NATIVE_FORMATTER_DECODER_COLUMN_BINDING",
    "R37_ONLY_MESSAGES_REQUIRES_DECODING","R37_REBUILT_NATIVE_FEATURE_CONFIG_DIFFERS","R37_EXACT_NATIVE_JSON_ARROW_ENCODING",
    "R37_EXACT_ACTIVE_VISITOR_FEATURE_AND_FORMATTER_METHODS","R37_REBUILT_FEATURE_ARROW_ENCODING_DIFFERS",
    "R37_EXPECTED_EXACT_NATIVE_LIST_REBUILD_UNOBSERVED","R37_EXPECTED_EXACT_NATIVE_JSON_LEAF_ORIGIN_UNOBSERVED",
    "R37_NATIVE_FEATURE_CONFIG_DIFFERS_FROM_FROZEN_DEFAULTS","R38_EXACT_PHYSICAL_IMAGE_STORAGE",
    "R38_EXACT_DECLARED_IMAGE_STORAGE_BOTH_LOADERS","R38_ORIGINAL_AND_NATIVE_IMAGE_PATH_DIFFER",
    "R38_NO_FEATURE_DECODING_FOR_STORAGE_READS","R35_NATIVE_WORK_CUTOFF","NATIVE_SAMPLER_CPU_DEADLINE",
    "JSON_DUPLICATE_OBJECT_KEY","JSON_NONFINITE_CONSTANT","EXACT_TWO_SOURCE_MESSAGE_RECORDS","EXACT_SOURCE_ROLE_AND_TEXT_TYPES",
    "EXACTLY_ONE_UNMODIFIED_MEDIA_ELEMENT","NO_DROPPED_OR_EMBEDDED_IMAGE_STORAGE","EXACT_ORIGINAL_TRAIN_MEDIA_PATH",
    "ARROW_NATIVE_DECODERS_DISAGREE",
}


def feature_identity(feature):
    qualified=type(feature).__module__+"."+type(feature).__qualname__
    return {"type_SHA256":type_id(feature),"qualified_type":qualified if re.fullmatch(r"[A-Za-z0-9_.]{1,160}",qualified) else None,
            "dataclass_fields":[field.name for field in dataclasses.fields(feature)] if dataclasses.is_dataclass(feature) else None}


def image_contract(projected,datasets,List,pa,report):
    typ=projected.schema.field("images").type
    observed={}
    for label,dataset in datasets:
        feature=dataset.features["images"]
        declared=dataset.features.arrow_schema.field("images")
        observed[label]={"outer_feature":feature_identity(feature),"nested_feature":feature_identity(getattr(feature,"feature",None)),
            "declared_type_SHA256":digest(encoded(str(declared.type))),"physical_type_matches":declared.type.equals(typ),
            "declared_shape":declared_shape(declared.type,pa.types)}
    report["image_feature_declarations"]=observed
    types=pa.types
    require(types.is_list(typ) and types.is_struct(typ.value_type),"R38_EXACT_PHYSICAL_IMAGE_STORAGE")
    struct=typ.value_type
    require(tuple(field.name for field in struct)==("bytes","path")
            and types.is_binary(struct.field("bytes").type) and types.is_string(struct.field("path").type),
            "R38_EXACT_PHYSICAL_IMAGE_STORAGE")
    require(all(type(dataset.features["images"]) is List and observed[label]["physical_type_matches"] is True for label,dataset in datasets),
            "R38_EXACT_DECLARED_IMAGE_STORAGE_BOTH_LOADERS")


def featureless_formatter(formatter_module):
    formatter=formatter_module.PythonFormatter(features=None,lazy=False)
    require(formatter.features is formatter.python_features_decoder.features is None,
            "R38_NO_FEATURE_DECODING_FOR_STORAGE_READS")
    return formatter


def safe_fault(report,index,inverse,stage,fault):
    if report["first_difference"] is not None:
        return None
    result={"physical_ordinal":index+1,"unshuffled_logical_ordinal":index+1,"shuffled_logical_ordinal":inverse[index]+1,
            "comparison_stage":stage,"error_type_SHA256":type_id(fault)}
    if type(fault) is ValueError and str(fault) in SAFE_CODES:
        result["bounded_rejection_class"]=str(fault)
    report["first_difference"]=result
    return result


def verify_message_pass(table,lines,inverse,datasets,pa,formatter_module,datasets_module,swift_core,report,summary,check):
    matched=set()
    columns=("messages",)
    try:
        projected=table.select(list(columns))
        check()
        pin_runtime(datasets_module,formatter_module,swift_core,summary)
        feature,native=native_message_reader(datasets[0][1],datasets[1][1],projected,pa,datasets_module,formatter_module,summary)
        storage=featureless_formatter(formatter_module)
    except ROW_FAILURES as error:
        safe_fault(report,0,inverse,"MESSAGE_FEATURE_SETUP",error)
        report["state"]="SETUP_BLOCKED"
        return matched
    for index,line in enumerate(lines):
        stage="MESSAGE_STORAGE_READERS"
        try:
            if not index%16:
                check()
            unit=projected.slice(index,1)
            current=decode(unit,storage,columns=columns)
            stage="MESSAGE_STRICT_NATIVE_AND_SOURCE"
            verify_json_message_values(unit,current["messages"],strict_json(line)["messages"],feature,native)
            matched.add(index)
        except ROW_FAILURES as error:
            preview=safe_fault(report,index,inverse,stage,error)
            if isinstance(error,TimeoutError) or type(error) is ValueError and str(error)=="R35_NATIVE_WORK_CUTOFF":
                report["state"]="INCOMPLETE_BUDGET"
                break
            if preview is not None:
                try:
                    unit=projected.slice(index,1)
                    values=decode(unit,storage,preview,columns)
                    if stage=="MESSAGE_STRICT_NATIVE_AND_SOURCE":
                        verify_json_message_values(unit,values["messages"],strict_json(line)["messages"],feature,native,preview)
                except ROW_FAILURES as diagnostic:
                    preview["decode_diagnostic_type_SHA256"]=type_id(diagnostic)
        report["rows_examined"]+=1
    report["qualified_rows"]=len(matched)
    report["qualified_elements"]=2*len(matched)
    if report["state"]!="INCOMPLETE_BUDGET":
        report["state"]="COMPLETE" if len(matched)==ROWS else "VALUE_OR_DECODER_MISMATCH"
    return matched


def verify_media_pass(table,lines,inverse,datasets,pa,formatter_module,List,prefix,report,check):
    matched=set()
    columns=("images",)
    try:
        projected=table.select(list(columns))
        check()
        image_contract(projected,datasets,List,pa,report)
        storage=featureless_formatter(formatter_module)
    except ROW_FAILURES as error:
        safe_fault(report,0,inverse,"MEDIA_FEATURE_AND_PHYSICAL_SCHEMA",error)
        report["state"]="SETUP_BLOCKED"
        return matched
    for index,line in enumerate(lines):
        stage="MEDIA_STORAGE_READERS"
        try:
            if not index%16:
                check()
            unit=projected.slice(index,1)
            current=decode(unit,storage,columns=columns)
            stage="MEDIA_EXACT_PATH_AND_NULL_BYTES"
            source=strict_json(line)
            require(exact(media(source["images"],"SOURCE",prefix),media(current["images"],"NATIVE_IMAGE_PATH_WITH_NULL_STORAGE",prefix)),
                    "R38_ORIGINAL_AND_NATIVE_IMAGE_PATH_DIFFER")
            matched.add(index)
        except ROW_FAILURES as error:
            preview=safe_fault(report,index,inverse,stage,error)
            if isinstance(error,TimeoutError) or type(error) is ValueError and str(error)=="R35_NATIVE_WORK_CUTOFF":
                report["state"]="INCOMPLETE_BUDGET"
                break
            if preview is not None:
                try:
                    decode(projected.slice(index,1),storage,preview,columns)
                except ROW_FAILURES as diagnostic:
                    preview["decode_diagnostic_type_SHA256"]=type_id(diagnostic)
        report["rows_examined"]+=1
    report["qualified_rows"]=len(matched)
    if report["state"]!="INCOMPLETE_BUDGET":
        report["state"]="COMPLETE" if len(matched)==ROWS else "VALUE_OR_DECODER_MISMATCH"
    return matched


def empty_pass(messages=False):
    record={"state":"NOT_REACHED","planned_rows":ROWS,"rows_examined":0,"qualified_rows":0,"first_difference":None}
    if messages:
        record.update(planned_elements=2*ROWS,qualified_elements=0)
    return record


def verify(raw,unshuffled,shuffled,Dataset,pa,formatter_module,datasets_module,swift_core,prefix,report,check):
    messages=empty_pass(messages=True)
    images=empty_pass()
    report.update(qualified_rows=0,planned_rows=ROWS,planned_message_elements=2*ROWS,qualified_message_elements=0,
                  qualified_media_rows=0,planned_media_rows=ROWS,message_pass=messages,media_pass=images,
                  runtime_pins_verified=False,permutation_verified=False,matched_complete=False,first_difference=None,
                  validation_stage="DATASET_AND_PHYSICAL_EQUIVALENCE")
    require(type(unshuffled) is type(shuffled) is Dataset and len(unshuffled)==len(shuffled)==ROWS,
            "EXACT_R35_NATIVE_DATASET_DENOMINATORS")
    first,second=unshuffled._data.table,shuffled._data.table
    require(len(first)==len(second)==ROWS and first.equals(second,check_metadata=True) and unshuffled._indices is None,
            "EXACT_R35_PHYSICAL_TABLE_IDENTITY")
    require(all(item["state"]=="AVAILABLE" for item in report["loader_metadata"].values()) and len(report["loader_metadata"])==2,
            "BOTH_LOADER_METADATA_REQUIRED")
    report["validation_stage"]="SHUFFLED_INDEX_PERMUTATION"
    require(shuffled._indices is not None and shuffled._indices.table.column_names==["indices"],"EXACT_R35_NATIVE_INDEX_COLUMN")
    order=shuffled._indices.table.column("indices").to_pylist()
    require(len(order)==ROWS and all(type(x) is int for x in order) and sorted(order)==list(range(ROWS)) and order!=list(range(ROWS)),
            "EXACT_R35_COMPLETE_NONIDENTITY_PERMUTATION")
    report["permutation_verified"]=True
    inverse={physical:logical for logical,physical in enumerate(order)}
    lines=raw.splitlines()
    require(len(lines)==ROWS,"R35_EXACT_SOURCE_800_ROWS")
    datasets=(("swift_unshuffled_call",unshuffled),("swift_shuffled_call",shuffled))
    report["validation_stage"]="INDEPENDENT_MESSAGE_AND_MEDIA_VALUES"
    message_matches=verify_message_pass(first,lines,inverse,datasets,pa,formatter_module,datasets_module,swift_core,messages,report,check)
    image_matches=verify_media_pass(first,lines,inverse,datasets,pa,formatter_module,datasets_module.List,prefix,images,check)
    combined=len(message_matches & image_matches)
    report.update(qualified_rows=combined,qualified_message_elements=2*len(message_matches),qualified_media_rows=len(image_matches))
    differences=[item["first_difference"] for item in (messages,images) if item["first_difference"] is not None]
    report["first_difference"]=min(differences,key=lambda item:item["physical_ordinal"],default=None)
    require(combined==ROWS and messages["state"]==images["state"]=="COMPLETE",
            "R38_INDEPENDENT_MESSAGE_AND_MEDIA_BOTH_REQUIRED")
    report["matched_complete"]=True
    report["validation_stage"]="COMPLETE"
    return [value+1 for value in order]
