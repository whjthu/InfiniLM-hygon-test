def load():
 return {'episodes':[{'label':n,'chunked':False,'deferred_feedback':True,'max_live':1,'n_requests':4,'arrival_policy':'all_zero','observer':v}for n,v in [('C1-OFF-before',False),('C1-ON-profile',True),('C1-OFF-after',False)]]}
