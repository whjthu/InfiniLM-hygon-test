"""Train-only schema and complete native supervision checks, no model imports."""


def check_record(row):
    if (not isinstance(row,dict) or type(row.get("id")) is not str or not row["id"]
            or len(row.get("messages",[]))!=2
            or [m.get("role") for m in row["messages"]]!=["user","assistant"]
            or type(row["messages"][1].get("content")) is not str
            or not row["messages"][1]["content"].startswith("FINAL: ")
            or len(row.get("images",[]))!=1 or type(row["images"][0]) is not str):
        raise ValueError("Original train-only message/image interface differs")


def check_encoded(ids,labels,eos,suffix_ids):
    if (type(eos) is not int or not 1<len(ids)==len(labels)<=4096
            or any(type(x) is not int or x<0 for x in ids)
            or any(type(x) is not int or (x<0 and x!=-100) for x in labels)
            or not suffix_ids or len(suffix_ids)>len(ids)
            or any(type(x) is not int or x<0 for x in suffix_ids)
            or eos not in suffix_ids or ids[-len(suffix_ids):]!=list(suffix_ids)):
        raise ValueError("Full native length, token type or supervised EOS differs")
    supervised=[i for i,x in enumerate(labels) if x!=-100]
    if not supervised or supervised[0]<=0 or any(labels[i]!=ids[i] for i in supervised):
        raise ValueError("Native prompt/answer supervision mask differs")
    suffix_start=len(ids)-len(suffix_ids)
    if not any(token==eos and labels[suffix_start+i]==eos for i,token in enumerate(suffix_ids)):
        raise ValueError("Actual native suffix has no supervised EOS")
    return dict(length=len(ids),supervised_tokens=len(supervised),
                shifted_supervised_tokens=sum(x!=-100 for x in labels[1:]))
