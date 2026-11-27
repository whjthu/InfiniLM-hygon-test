"""Local mapping and packet fixtures. Target Swift and Arrow are absent on Windows."""
import json
from types import SimpleNamespace
import unittest

import lineage as q
import native_route
import prepare


class Column:
    def __init__(self,values):
        self.values=values

    def __getitem__(self,i):
        return SimpleNamespace(as_py=lambda:self.values[i])

    def to_pylist(self):
        return self.values


class Table:
    def __init__(self,cols):
        self.cols=cols
        self.column_names=list(cols)
        self.num_rows=len(next(iter(cols.values())))

    def column(self,name):
        return Column(self.cols[name])

    def equals(self,other,check_metadata=False):
        return check_metadata and self.cols==other.cols


class Dataset:
    def __init__(self,table,order=None):
        self._data=SimpleNamespace(table=table)
        self._indices=SimpleNamespace(table=Table(dict(indices=order))) if order is not None else None

    def __len__(self):
        return self._data.table.num_rows

    def select_columns(self,columns):
        assert columns==["messages"]
        return self

    def with_format(self,fmt):
        assert fmt is None
        return self

    def __getitem__(self,i):
        if self._indices:
            i=self._indices.table.cols["indices"][i]
        raw=self._data.table.cols["messages"][i]
        return dict(messages=q.shape_messages(raw))


class Fixture(unittest.TestCase):
    def fixture(self):
        source=list(range(1,57))+list(range(1,9))
        rows=[dict(messages=[dict(role="user",content=f"synthetic-{index}"),
                             dict(role="assistant",content="synthetic")],
                   images=[f"/train/{index}.jpg"]) for index in source]
        lines=[q.encode(row).rstrip(b"\n") for row in rows]
        proof={index:q.digest(raw) for index,raw in zip(source,lines,strict=True)}
        slots=[dict(source_ordinal=index,row_SHA256=proof[index],repeat=physical>=56)
               for physical,index in enumerate(source)]
        expected=q.freeze_physical(b"\n".join(lines),slots,proof,set(proof))
        table=Table(dict(messages=[[json.dumps(item) for item in row["messages"]] for row in rows],
                         images=[[dict(bytes=None,path=row["images"][0])] for row in rows]))
        order=list(range(63,-1,-1))
        return expected,table,order

    def test_repeat_values_are_distinct_occurrences_from_actual_fetch(self):
        expected,table,order=self.fixture()
        shuffle,messages,values=q.arrow_projection(Dataset(table),Dataset(table,order),Dataset,expected)
        dataset=q.ObservedOccurrences(messages,table,shuffle)
        rank_indices=[list(range(rank,64,8)) for rank in range(8)]
        reports=[q.verify_rank(rank,[q.collate_observed([dataset[index]]) for index in indexes],
                               rank_indices,shuffle,expected,values) for rank,indexes in enumerate(rank_indices)]
        result=q.combine(reports,set(range(1,57)))
        self.assertEqual((result['physical_occurrences'],result['unique_source_rows'],
                          result['repeat_occurrences'],result['internal_source_rows']),(64,56,8,0))
        self.assertEqual(values[0],values[56])
        self.assertNotEqual(dataset[63]['physical'],dataset[7]['physical'])
        self.assertEqual(expected[0][1]['source_ordinal'],expected[56][1]['source_ordinal'])
        wrong=[dataset[index] for index in rank_indices[0]]
        wrong[0],wrong[1]=wrong[1],wrong[0]
        with self.assertRaisesRegex(ValueError,"RANK_OCCURRENCE_DRIFT"):
            q.verify_rank(0,wrong,rank_indices,shuffle,expected,values)

    def test_exact_physical_mapping_rejects_duplicate_non_null_and_changed_message(self):
        expected,table,order=self.fixture()
        with self.assertRaisesRegex(ValueError,"64_PHYSICAL_OCCURRENCES"):
            q.arrow_projection(Dataset(table),Dataset(table,[*order[:-1],order[0]]),Dataset,expected)
        table.cols["images"][0][0]["bytes"]=b"body"
        with self.assertRaisesRegex(ValueError,"NULL_MEDIA_BYTES"):
            q.arrow_projection(Dataset(table),Dataset(table,order),Dataset,expected)
        table.cols["images"][0][0]["bytes"]=None
        table.cols["messages"][0][1]='{"role":"assistant","content":"x","content":"y"}'
        with self.assertRaisesRegex(ValueError,"DUPLICATE_JSON_KEY"):
            q.arrow_projection(Dataset(table),Dataset(table,order),Dataset,expected)

    def test_frozen_packet_uses_original_production_cli(self):
        files=prepare.packet()
        self.assertEqual(prepare.packet(),files)
        cmd=json.loads(files["original-launch.json"])["command"]
        args=native_route.rewrite_args(cmd,"/data/train","/model-metadata","/unused-output")
        expected={"--dataset":"/data/train","--dataset_shuffle":"true","--dataloader_num_workers":"4",
                  "--data_seed":"42","--seed":"42","--max_steps":"25","--split_dataset_ratio":"0"}
        self.assertTrue(all(args[args.index(key)+1]==value for key,value in expected.items()))
        self.assertNotIn("nvidia_baseline",args)


if __name__=="__main__":
    unittest.main()
