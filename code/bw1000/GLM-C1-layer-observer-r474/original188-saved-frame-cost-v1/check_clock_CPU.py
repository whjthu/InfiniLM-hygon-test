"""New synthetic clocks only; this does not read any original GLM artifact."""
import json
from costs import interval, overlap


def main():
    marks = [dict(owner_host_s=x+100, gpu_from_first_s=x) for x in (0,1,2,3)]
    outer=dict(clock_identity=[0,0,1],**interval(marks,0,3))
    inner=dict(clock_identity=[0,0,1],**interval(marks,1,2))
    assert outer['host_envelope_s']==outer['event_envelope_s']==3
    assert overlap(outer,inner,'host')==overlap(outer,inner,'event')==1
    rejected=[]
    def bad(name,call):
        try:
            call()
        except ValueError:
            rejected.append(name)
        else:
            raise AssertionError(name)
    bad('bad_unit',lambda:interval(marks,0,3,{'owner_host_s':'milliseconds','gpu_from_first_s':'seconds'}))
    inverted=[dict(owner_host_s=2,gpu_from_first_s=0),dict(owner_host_s=1,gpu_from_first_s=1)]
    bad('negative_host',lambda:interval(inverted,0,1))
    inverted=[dict(owner_host_s=0,gpu_from_first_s=2),dict(owner_host_s=1,gpu_from_first_s=1)]
    bad('negative_event',lambda:interval(inverted,0,1))
    bad('missing_marker',lambda:interval(marks,0,4))
    foreign=dict(inner,clock_identity=[1,0,1])
    bad('cross_rank_clock',lambda:overlap(outer,foreign,'event'))
    rounding=[dict(owner_host_s=0,gpu_from_first_s=0.1),dict(owner_host_s=1,gpu_from_first_s=.1-1e-7)]
    assert interval(rounding,0,1)['event_envelope_s'] is None
    print(json.dumps(dict(normal=1,nested_overlap=1,event_rounding_UNKNOWN=1,rejected=rejected,
                          Native=False,GPU=False,original_artifact_IO=False)))


if __name__=='__main__':
    main()
