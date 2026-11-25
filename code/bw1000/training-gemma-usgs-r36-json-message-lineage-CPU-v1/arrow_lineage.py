"""Describe both actual providers, then compare source and Arrow in physical order."""

from hashlib import sha256
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


def image_shape(datatype,feature,types):
    if list_like(datatype,types):
        child=datatype.value_type
        if text_like(child,types):
            return "TEXT_PATH_LIST"
        leaf=getattr(feature,"feature",None)
        if (struct_fields(child,types,("bytes","path")) and text_like(child.field("path").type,types)
            and (types.is_binary(child.field("bytes").type) or types.is_large_binary(child.field("bytes").type))
            and (type(leaf).__module__,type(leaf).__name__)==("datasets.features.image","Image")):
            return "NATIVE_IMAGE_PATH_WITH_NULL_STORAGE"
    raise ValueError("UNPROVEN_ARROW_IMAGE_LAYOUT")


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


def decode(unit,formatter,evidence=None):
    scalars={name:unit.column(name)[0] for name in COLS}
    if evidence is not None:
        evidence["scalar_classes"]={k:{"type_SHA256":type_id(v),"declared_type_SHA256":type_id(v.type),"valid":v.is_valid} for k,v in scalars.items()}
    scalar={name:value.as_py() for name,value in scalars.items()}
    if evidence is not None:
        evidence["native_scalar_as_py"]={k:shapes(v) for k,v in scalar.items()}
    table,=unit.to_pylist()
    if evidence is not None:
        evidence["native_arrow_table_return"]={k:shapes(v) for k,v in table.items() if k in COLS}
    formatted=formatter.format_row(unit)
    if evidence is not None:
        evidence["native_nonmedia_formatter_return"]={k:shapes(v) for k,v in formatted.items() if k in COLS}
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


def native_message_reader(first,second,projected,pa,datasets_module,formatter_module):
    feature=first.features["messages"]
    paired=second.features["messages"]
    Json,List=datasets_module.Json,datasets_module.List
    typ=projected.schema.field("messages").type
    expected=pa.json_()
    require(type(feature) is type(paired) is List and type(feature.feature) is type(paired.feature) is Json
            and feature.feature.decode is paired.feature.decode is True and feature.feature.pa_type.equals(expected)
            and paired.feature.pa_type.equals(expected) and pa.types.is_list(typ) and typ.value_type.equals(expected)
            and typ.equals(first.features.arrow_schema.field("messages").type)
            and typ.equals(second.features.arrow_schema.field("messages").type),"R36_EXACT_ACTIVE_NATIVE_LIST_OF_JSON_FEATURE")
    scoped=datasets_module.Features({"messages":feature})
    native=formatter_module.PythonFormatter(features=scoped,lazy=False)
    require(set(native.features)=={"messages"} and native.features["messages"] is feature,
            "R36_FEATURE_DECODING_RESTRICTED_TO_MESSAGES")
    return feature.feature,native


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


def verify(raw,unshuffled,shuffled,Dataset,pa,formatter_module,datasets_module,swift_core,prefix,report,check):
    report.update(qualified_rows=0,planned_rows=ROWS,planned_message_elements=2*ROWS,qualified_message_elements=0,
                  runtime_pins_verified=False,permutation_verified=False,matched_complete=False,first_difference=None)
    report["validation_stage"]="DATASET_AND_PHYSICAL_EQUIVALENCE"
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
    projected=first.select(list(COLS))
    formatter=formatter_module.PythonFormatter(features=None,lazy=False)
    require(formatter.features is None,"R35_MEDIA_FEATURE_DECODING_OFF")
    lines=raw.splitlines()
    require(len(lines)==ROWS,"R35_EXACT_SOURCE_800_ROWS")
    providers=None
    runtime_verified=False
    report["validation_stage"]="PER_ROW_EXACT_EQUIVALENCE"
    for index,line in enumerate(lines):
        if not index%16:
            check()
        comparison="STRUCTURE"
        try:
            if providers is None:
                pin_runtime(datasets_module,formatter_module,swift_core,report)
                runtime_verified=True
                providers=native_message_reader(unshuffled,shuffled,projected,pa,datasets_module,formatter_module)
                require(all(projected.schema.field("images").type.equals(dataset.features.arrow_schema.field("images").type)
                            and image_shape(projected.schema.field("images").type,dataset.features["images"],pa.types)
                            =="NATIVE_IMAGE_PATH_WITH_NULL_STORAGE" for dataset in (unshuffled,shuffled)),
                        "R36_EXACT_IMAGE_PATH_NULL_BYTES_STORAGE")
                report["declared_lossless_layouts"]=["NATIVE_LIST_JSON_EXTENSION","NATIVE_IMAGE_PATH_WITH_NULL_STORAGE"]
            unit=projected.slice(index,1)
            comparison="FEATURELESS_NATIVE_STORAGE"
            current=decode(unit,formatter)
            source=strict_json(line)
            comparison="STRICT_NATIVE_JSON_AND_SOURCE"
            verify_json_message_values(unit,current["messages"],source["messages"],*providers)
            comparison="EXACT_PATH_NULL_IMAGE_BYTES"
            require(exact(media(source["images"],"SOURCE",prefix),
                          media(current["images"],"NATIVE_IMAGE_PATH_WITH_NULL_STORAGE",prefix)),
                    "R36_EXACT_ORIGINAL_IMAGE_PATH")
        except (OSError,ValueError,KeyError,TypeError,AttributeError,RuntimeError,IndexError) as fault:
            preview={"physical_ordinal":index+1,"unshuffled_logical_ordinal":index+1,"shuffled_logical_ordinal":inverse[index]+1,
                     "comparison_stage":comparison,"error_type_SHA256":type_id(fault)}
            if type(fault) is ValueError and re.fullmatch(r'[A-Z0-9_]{1,100}',str(fault)):
                preview["bounded_rejection_class"]=str(fault)
            report["first_difference"]=preview
            try:
                unit=projected.slice(index,1)
                captured=decode(unit,formatter,preview)
                if runtime_verified and providers is not None and comparison=="STRICT_NATIVE_JSON_AND_SOURCE":
                    verify_json_message_values(unit,captured["messages"],strict_json(line)["messages"],*providers,preview)
            except (OSError,ValueError,KeyError,TypeError,AttributeError,RuntimeError,IndexError) as diagnostic:
                preview["decode_diagnostic_type_SHA256"]=type_id(diagnostic)
            raise ValueError("R35_ARROW_LOSSLESS_SOURCE_CORRESPONDENCE_UNPROVEN") from None
        report["qualified_rows"]+=1
        report["qualified_message_elements"]+=2
    report["matched_complete"]=True
    report["validation_stage"]="COMPLETE"
    return [value+1 for value in order]
