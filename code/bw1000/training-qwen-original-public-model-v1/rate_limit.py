"""Bound only this new public-model relay; never alter an existing transfer."""
import math
def delay_after_chunk(copied,elapsed,rate,remaining):
 if type(copied) is not int or copied<0 or type(rate) is not int or rate!=1048576:raise ValueError('invalid public-model rate contract')
 if not math.isfinite(elapsed) or elapsed<0 or not math.isfinite(remaining) or remaining<=0:raise ValueError('invalid original rate clock')
 delay=max(0.,copied/rate-elapsed)
 if delay>=remaining-15:raise TimeoutError('original rate budget cannot retain closing reserve')
 return delay
