import pathlib
p=pathlib.Path(__file__).resolve().parent
for n in ['modern_glm_bridge_observer_r131.cpp','glm_text_observer_r131.cpp','glm_decoder_observer_r131.cpp','glm_prepare_observer_r131.cpp']:
 (p/n).unlink();(p/(n+'.patch')).unlink()
exec(compile((p/'prepare_sources_r131.py').read_text(encoding='utf8'),str(p/'prepare_sources_r131.py'),'exec'))
