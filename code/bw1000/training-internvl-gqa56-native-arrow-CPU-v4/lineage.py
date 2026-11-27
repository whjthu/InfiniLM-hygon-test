"""Join observed Swift Arrow physical occurrences to the frozen InternVL training roster."""
from collections import Counter
from hashlib import sha256
import json


N=64
WORLD=8
LOGICAL_REFERENCE_SHA="639aeef07c7e916f1e3bc49a02ac9637547c2c654ae10e5c997e4c0bc81b43f8"
SAMPLER_RESULT_SHA="60599327b076a7729331741927cf4d05be4927f086ece691b4c7c4fbe2545cd5"
SAMPLER_REQUEST="9056f6af43864cf4934af5026451d063"


def need(value,code):
    if not value:
        raise ValueError(code)


def encode(value):
    return (json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False)+"\n").encode()


def digest(value):
    return sha256(value).hexdigest()


def strict_json(raw):
    def object_items(items):
        obj=dict(items)
        need(len(obj)==len(items),"DUPLICATE_JSON_KEY")
        return obj

    def bad_number(_):
        raise ValueError("NONFINITE_JSON_CONSTANT")

    return json.loads(raw,object_pairs_hook=object_items,parse_constant=bad_number)


def check_permutation(values):
    need(type(values) is list and len(values)==N and all(type(x) is int for x in values) and
         set(values)==set(range(N)),"ALL_64_PHYSICAL_OCCURRENCES_REQUIRED")
    return values


def check_matrix(rows):
    need(type(rows) is list and len(rows)==WORLD and all(type(row) is list and len(row)==8 for row in rows),
         "ALL_EIGHT_RANK_ROWS_REQUIRED")
    check_permutation([x for row in rows for x in row])
    return rows


def reference_matrix(raw):
    need(digest(raw)==SAMPLER_RESULT_SHA,"ORIGINAL_4643_RESULT_BYTES")
    receipt=strict_json(raw)
    logical=check_matrix(receipt["rank_logical_indices"])
    need(receipt["request"]==SAMPLER_REQUEST and receipt["actual_step"]=="1103606.4643" and
         receipt["normal_ranks"]==receipt["reaped_ranks"]==8 and not receipt["survivor_ranks"] and
         receipt["native_logical_permutation_verified"] is True and
         receipt["local_filestore_cleanup"]["state"]=="RETIRED" and
         digest(encode(logical))==receipt["native_order_SHA256"]==LOGICAL_REFERENCE_SHA,
         "ORIGINAL_4643_EIGHT_RANK_CLEAN_ORDER")
    return logical


def shape_messages(value):
    need(type(value) is list and len(value)==2,"EXACT_TWO_MESSAGES")
    if all(type(item) is str for item in value):
        value=[strict_json(item) for item in value]
    need(all(type(item) is dict and set(item)=={"role","content"} and
             type(item["content"]) is str and item["content"] for item in value) and
         [item["role"] for item in value]==["user","assistant"],"EXACT_MESSAGES_VALUE_TYPE")
    return value


def shape_images(value):
    need(type(value) is list and len(value)==1,"EXACT_ONE_MEDIA_MEMBER")
    image=value[0]
    if type(image) is dict:
        need(set(image)=={"bytes","path"} and image["bytes"] is None,"EXACT_PATH_ONLY_NULL_MEDIA_BYTES")
        image=image["path"]
    need(type(image) is str and bool(image),"EXACT_ORIGINAL_MEDIA_PATH_TYPE")
    return [image]


def canonical_row(value):
    need(type(value) is dict and set(value)=={"messages","images"},"EXACT_TRAIN_ROW_COLUMNS")
    return dict(messages=shape_messages(value["messages"]),images=shape_images(value["images"]))


def freeze_physical(exposure_raw, slots, roster, train_only):
    """Use the previously bound original roster and exact physical slot manifest."""
    lines=exposure_raw.splitlines()
    need(len(lines)==len(slots)==64 and len(roster)==56 and len(train_only)==56,"EXACT_FROZEN_56_PLUS_8_BOUNDARY")
    expected={}
    counts=Counter()
    for physical,(raw,slot) in enumerate(zip(lines,slots,strict=True)):
        current=digest(raw)
        ordinal=slot["source_ordinal"]
        need(type(ordinal) is int and ordinal in train_only and type(slot["repeat"]) is bool and
             roster.get(ordinal)==current==slot["row_SHA256"] and slot["repeat"]==(physical>=56),
             "EXACT_FROZEN_PHYSICAL_4370_SOURCE")
        occurrence=dict(physical=physical,source_ordinal=ordinal,row_SHA256=current)
        expected[physical]=(canonical_row(strict_json(raw)),occurrence)
        counts[ordinal]+=1
    need(set(counts)==set(train_only) and Counter(counts.values())=={1:48,2:8},
         "EXACT_56_UNIQUE_AND_EIGHT_REPEAT_OCCURRENCES")
    return expected


def arrow_projection(unshuffled,shuffled,dataset_type,expected,dataset_source):
    """Observe native raw cells and the datasets formatter, without opening images."""
    need(type(unshuffled) is type(shuffled) is dataset_type and len(unshuffled)==len(shuffled)==64 and
         unshuffled._indices is None and shuffled._indices is not None,"EXACT_TWO_NATIVE_ARROW_LOADS")
    first,second=unshuffled._data.table,shuffled._data.table
    need(first.num_rows==second.num_rows==64,"EXACT_BOTH_ARROW_PHYSICAL_ROW_COUNTS")
    need(all(len(table.column_names)==3 and set(table.column_names)=={"messages","images","dataset"}
             for table in (first,second)),"EXACT_BOTH_ARROW_PHYSICAL_COLUMNS")
    need(first.schema.equals(second.schema,check_metadata=False),"EXACT_SAME_ARROW_PHYSICAL_SCHEMA")
    need(first.schema.equals(second.schema,check_metadata=True),"EXACT_SAME_ARROW_SCHEMA_METADATA")
    need(type(dataset_source) is str and dataset_source.startswith("/"),"EXACT_FROZEN_ARROW_ROUTE_ABSOLUTE_PATH")
    for table in (first,second):
        need(str(table.schema.field("dataset").type)=="string","EXACT_ARROW_DATASET_ROUTE_STRING_FIELD")
        column=table.column("dataset")
        for index in range(N):
            route=column[index].as_py()
            need(type(route) is str,"EXACT_ARROW_DATASET_ROUTE_TYPED_CELL")
            need(route==dataset_source,"EXACT_ARROW_DATASET_ROUTE_BOUND_EXPOSURES_PATH")
    need(first.equals(second,check_metadata=False),"EXACT_SAME_ARROW_PHYSICAL_CELL_VALUES")
    ind=shuffled._indices.table
    need(ind.column_names==["indices"],"EXACT_NATIVE_ARROW_INDEX_COLUMN")
    logical_to_physical=check_permutation(ind.column("indices").to_pylist())
    need(logical_to_physical!=list(range(N)),"NATIVE_SHUFFLE_WAS_NOT_OBSERVED")
    # Select message columns only before invoking native __getitem__; image decode is excluded.
    native_first=unshuffled.select_columns(["messages"]).with_format(None)
    native_second=shuffled.select_columns(["messages"]).with_format(None)
    inverse={physical:logical for logical,physical in enumerate(logical_to_physical)}
    row_digests={}
    for physical,(wanted,occurrence) in expected.items():
        raw={key:first.column(key)[physical].as_py() for key in ("messages","images")}
        strict=canonical_row(raw)
        plain=native_first[physical]["messages"]
        moved=native_second[inverse[physical]]["messages"]
        need(all(type(records) is list and all(type(item) is dict for item in records)
                 for records in (plain,moved)) and strict==wanted and
             shape_messages(plain)==shape_messages(moved)==wanted["messages"],
             "EXACT_ARROW_PHYSICAL_AND_LOGICAL_ROW_VALUES")
        # This digest represents the actual typed values, separately from original JSONL bytes.
        row_digests[physical]=digest(encode(strict))
    return logical_to_physical,native_second,row_digests


class ObservedOccurrences:
    """Metadata ablation for the real DataLoaderShard; never an image/model collator."""

    def __init__(self, messages, table, logical_to_physical):
        self.messages=messages
        self.table=table
        self.logical_to_physical=tuple(logical_to_physical)

    def __len__(self):
        return len(self.messages)

    def __getitem__(self,logical):
        need(type(logical) is int and 0<=logical<len(self),"EXACT_NATIVE_LOGICAL_INDEX")
        physical=self.logical_to_physical[logical]
        message=self.messages[logical]["messages"]
        need(type(message) is list and all(type(item) is dict for item in message),"NATIVE_OBSERVED_MESSAGES_REQUIRED")
        value=canonical_row(dict(messages=message,images=self.table.column("images")[physical].as_py()))
        return dict(logical=logical,physical=physical,typed_row_SHA256=digest(encode(value)))


def collate_observed(rows):
    need(type(rows) is list and len(rows)==1 and set(rows[0])=={"logical","physical","typed_row_SHA256"},
         "EXACT_METADATA_COLLATOR_BATCH")
    return rows[0]


def verify_rank(rank, observed, reference, order, expected, actual_value_hashes):
    need(type(rank) is int and 0<=rank<WORLD and len(observed)==8,"EXACT_EIGHT_LOADER_SHARD_SLOTS")
    result=[]
    for offset,(item,want) in enumerate(zip(observed,reference[rank],strict=True)):
        physical=order[want]
        need(item==dict(logical=want,physical=physical,typed_row_SHA256=actual_value_hashes[physical]),
             "ACTUAL_LOADER_RANK_OCCURRENCE_DRIFT")
        occurrence=expected[physical][1]
        result.append(dict(rank=rank,update=offset//4+1,accumulation=offset%4+1,logical=want,**occurrence))
    return result


def combine(observed,train_only):
    need(len(observed)==WORLD and all(len(row)==8 for row in observed),"EXACT_FULL_PRODUCTION_RANK_DENOMINATOR")
    rows=[item for values in observed for item in values]
    check_permutation([item["physical"] for item in rows])
    counts=Counter(item["source_ordinal"] for item in rows)
    need(set(counts)==set(train_only) and Counter(counts.values())=={1:48,2:8},
         "EXACT_NATIVE_56_UNIQUE_EIGHT_REPEAT_ZERO_INTERNAL")
    return dict(physical_occurrences=64,unique_source_rows=56,repeat_occurrences=8,internal_source_rows=0,
                per_update_unique_sources=[len({row["source_ordinal"] for row in rows if row["update"]==i})
                                           for i in (1,2)],
                actual_rank_occurrence_SHA256=digest(encode(observed)))
