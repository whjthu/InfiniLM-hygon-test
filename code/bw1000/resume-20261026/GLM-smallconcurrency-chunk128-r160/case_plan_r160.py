def load():
 small=[{'label':label,'chunked':False,'deferred_feedback':True,'max_live':C,'n_requests':8,'arrival_policy':'all_zero'}for label,C in [('C1-first8',1),('C8-first8',8)]]
 pair=[{'label':label,'chunked':chunked,'deferred_feedback':True,'max_live':32,'n_requests':32,'arrival_policy':'original32'}for label,chunked in [('A1-full',False),('B1-chunk128',True),('B2-chunk128',True),('A2-full',False)]]
 return {'episodes':small+pair,'small_scope':'First8 identical unmodified QA; finite queue, no row recycling, Native shape changes naturally with admission; not HTTP/continuous service','chunk_scope':'Same32 original arrivals0/4/8/12; full-prefix vs chunk128 preferdecode; deferON both'}
