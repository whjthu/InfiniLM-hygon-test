"""Source/value/split/QA contracts; stdlib only, no network or rendering imports."""
from decimal import Decimal,ROUND_HALF_UP
import hashlib
import json
from pathlib import Path
import re


def need(ok,why):
    if not ok:raise ValueError(why)


def canonical(value):return (json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False)+'\n').encode()
def sha(value):return hashlib.sha256(canonical(value)).hexdigest()


def strict_json(raw):
    def object_pairs(items):
        need(len(items)==len({k for k,v in items}),'duplicate JSON key');return dict(items)
    def constant(value):raise ValueError('nonfinite JSON number')
    return json.loads(raw,parse_float=Decimal,parse_int=int,parse_constant=constant,object_pairs_hook=object_pairs)


def plan_check(plan):
    need(plan['source_id']=='2' and plan['value_cutoff_year']==2020,'wrong source/cutoff')
    all_countries=[];all_years=[]
    for split,group in plan['splits'].items():
        need(split in ('train','dev','blind') and len(group['countries'])==8 and len(set(group['countries']))==8
            and len(set(group['strata'].values()))==8,'country/stratum quota mismatch')
        need(group['years']==list(range(group['years'][0],group['years'][0]+7)),'not exact7 consecutive annual dates')
        all_countries.extend(group['countries']);all_years.extend(group['years'])
    need(len(all_countries)==len(set(all_countries))==24 and len(all_years)==len(set(all_years))==21,'country/time leakage between splits')
    for stratum,countries in plan['strata'].items():
        order=sorted(countries,key=lambda c:hashlib.sha256((plan['selection_seed']+':'+stratum+':'+c).encode()).hexdigest())
        for split,country in zip(('train','dev','blind'),order):
            need(country in plan['splits'][split]['countries'] and plan['splits'][split]['strata'][country]==stratum,'Frozen seed/group assignment differs')
    need(len(plan['indicators'])==3 and plan['planned_charts']==72 and plan['planned_QA']==648 and plan['planned_facts']==504,'planned denominator differs')
    need(sum(e['cap_bytes'] for e in plan['endpoints'].values())<=8388608,'HTTP cap exceeds8MiB')
    need(72*plan['max_chart_bytes']<=33554432,'planned charts exceed32MiB')
    for split,group in plan['splits'].items():
        expected='https://api.worldbank.org/v2/country/'+ ';'.join(sorted(group['countries']))+'/indicator/'+ ';'.join(sorted(plan['indicators']))+'?source=2&date='+str(group['years'][0])+':'+str(group['years'][-1])+'&format=json&per_page=1000&page=1&footnote=y'
        need(plan['endpoints']['facts_'+split]['URL']==expected,'Frozen API query/split differs')
    return True


def array_response(payload,expected_total):
    need(isinstance(payload,list) and len(payload)==2 and isinstance(payload[0],dict) and isinstance(payload[1],list),'not official API response array')
    head,rows=payload
    need(head.get('page')==head.get('pages')==1 and int(head.get('total',-1))==expected_total
        and len(rows)==expected_total,'missing/extra/multipage official response')
    return head,rows


def parse_indicator(payload,code,plan):
    _,rows=array_response(payload,1);row=rows[0]
    need(row['id']==code and row['name']==plan['indicators'][code]['name']
        and row['source']['id']=='2' and row['source']['value']=='World Development Indicators'
        and row['unit']=='' and isinstance(row['sourceNote'],str) and row['sourceNote'].strip()
        and isinstance(row['sourceOrganization'],str) and row['sourceOrganization'].strip(),'indicator schema/meaning/source changed')
    # Metadata rights overrides are never silently discarded.
    rights={k:v for k,v in row.items() if re.search('licen|copyright|restriction|rights',k,re.I)}
    need(not rights,'new explicit metadata rights statement requires source review')
    return dict(code=code,name=row['name'],API_unit_field=row['unit'],unit_label=plan['indicators'][code]['unit_label'],
        sourceNote=row['sourceNote'],sourceOrganization=row['sourceOrganization'],additional_rights_metadata=rights)


def country_names_from_facts(payloads,plan):
    countries={};counts={}
    need(set(payloads)=={'train','dev','blind'},'All three frozen fact splits required')
    for split,payload in payloads.items():
        _,rows=array_response(payload,168);group=plan['splits'][split]
        for row in rows:
            code=row['countryiso3code'];name=row['country']['value']
            need(code in group['countries'] and isinstance(name,str) and name.strip()==name and bool(name),'Nonempty consistent official country name required')
            need(code not in countries or countries[code]['name']==name,'Country name drift within source facts')
            countries[code]=dict(name=name,name_source='official fact row country.value');counts[code]=counts.get(code,0)+1
    expected={c for g in plan['splits'].values() for c in g['countries']}
    need(set(countries)==expected and all(counts[c]==21 for c in expected),'All24 country names require21 actual rows each')
    return countries


def parse_facts(payload,split,plan):
    group=plan['splits'][split];head,rows=array_response(payload,168)
    need(str(head.get('sourceid'))=='2' and re.fullmatch(r'\d{4}-\d{2}-\d{2}',head.get('lastupdated','')),'source/version stamp absent')
    cells={};missing=[]
    for row in rows:
        code=row['countryiso3code'];indicator=row['indicator']['id'];year=int(row['date'])
        need(code in group['countries'] and indicator in plan['indicators'] and year in group['years'],'facts outside frozen country/indicator/time split')
        key=(code,indicator,year);need(key not in cells,'duplicate source fact')
        need(row['unit']=='' and row['indicator']['value']==plan['indicators'][indicator]['name'],'fact unit/name changed')
        value=row['value']
        if value is None:
            cells[key]=None;missing.append(key);continue
        need(type(value) in (int,Decimal) and Decimal(value).is_finite() and abs(Decimal(value))<=Decimal('1000000'),'invalid/nonfinite/unbounded official numeric value')
        cells[key]=Decimal(value)
    expected={(c,i,y) for c in group['countries'] for i in plan['indicators'] for y in group['years']}
    need(set(cells)==expected,'source fact denominator incomplete')
    return cells,dict(sourceid='2',lastupdated=head['lastupdated'],planned=168,actual_nonmissing=168-len(missing),missing=len(missing))


def display(value):
    value=Decimal(value).quantize(Decimal('.01'),rounding=ROUND_HALF_UP)
    return format(value.copy_abs() if value==0 else value,'.2f')


def questions(country,indicator,years,values,chart_id,plan):
    need(len(years)==len(values)==7 and all(v is not None for v in values),'whole7value chart required; no imputation')
    rounded=[Decimal(display(v)) for v in values];items=[]
    def item(kind,index,text,answer,year_answer=False):
        identifier=chart_id+':'+kind+':'+str(index)
        request='Read the derived World Bank WDI chart. Values printed beside points are rounded to two decimal places. '+text
        request+=' Reply exactly FINAL: YYYY with the four-digit year.' if year_answer else ' Use only the displayed rounded values. Reply exactly FINAL: <number> with two decimal places; include a minus sign for negative values.'
        items.append(dict(id=identifier,kind=kind,question=request,answer=answer,answer_type='year' if year_answer else 'decimal2',country_group=country,chart_group=chart_id,indicator=indicator))
    for n,k in enumerate((0,3,6)):item('lookup',n,f'What is the displayed value for {years[k]}?',display(rounded[k]))
    for n,(a,b) in enumerate(((0,3),(3,6),(0,6))):
        item('change',n,f'What is the change from {years[a]} to {years[b]}, calculated as the later value minus the earlier value, in percentage points?',display(rounded[b]-rounded[a]))
    maximum=max(rounded);minimum=min(rounded)
    item('extrema',0,'What is the maximum displayed value across all seven years?',display(maximum))
    item('extrema',1,'What is the minimum displayed value across all seven years?',display(minimum))
    item('extrema',2,'In which year is the maximum displayed value reached? If tied, choose the earliest year.',str(years[rounded.index(maximum)]),True)
    return items,dict(displayed_values=[display(v) for v in rounded],numeric_vector_SHA256=sha([display(v) for v in rounded]))


def score(labels,outputs,expected=648):
    need(len(labels)==expected and len({r['id'] for r in labels})==expected,'full scoring label denominator differs')
    need(set(outputs)<=set(r['id'] for r in labels),'unknown model output IDs')
    correct=missing=parse_fail=data_fail=0;groups={};kinds={}
    for row in labels:
        ok=False;raw=outputs.get(row['id'])
        if row.get('data_status')!='QUALIFIED':data_fail+=1
        elif raw is None:missing+=1
        elif not isinstance(raw,str):parse_fail+=1
        else:
            pattern=r'FINAL: ([0-9]{4})' if row['answer_type']=='year' else r'FINAL: ([+-]?(?:0|[1-9][0-9]*)\.[0-9]{2})'
            match=re.fullmatch(pattern,raw.strip())
            if match is None:parse_fail+=1
            else:ok=(match[1]==row['answer']) if row['answer_type']=='year' else (Decimal(match[1])==Decimal(row['answer']))
        correct+=ok
        for target,key in ((groups,row['country_group']),(kinds,row['kind'])):
            target.setdefault(key,dict(correct=0,planned=0));target[key]['correct']+=ok;target[key]['planned']+=1
    return dict(planned=expected,correct=correct,missing_outputs=missing,parse_failures=parse_fail,data_failures=data_fail,
        wrong_answers=expected-correct-missing-parse_fail-data_fail,
        accuracy=correct/expected,country_groups=groups,question_kinds=kinds,quality_qualified=False,
        denominator_includes_all_failures=True,model_training_qualified=False)
