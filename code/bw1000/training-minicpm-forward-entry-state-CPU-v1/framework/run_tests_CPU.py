import io,json,torch,unittest
assert torch.cuda.device_count()==0 and not torch.cuda.is_initialized()
torch.set_num_threads(1);torch.set_num_interop_threads(1)
stream=io.StringIO();suite=unittest.TestLoader().loadTestsFromNames(['vlm.tests.test_forward_entry_state','vlm.tests.test_resume_witness','vlm.tests.test_resume_pair']);result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
value=dict(tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),skipped=len(result.skipped),passed=result.wasSuccessful(),GPU_cards=0,CUDA_initialized=torch.cuda.is_initialized(),torch_version=torch.__version__,CPU_threads=torch.get_num_threads(),test_output=stream.getvalue())
print(json.dumps(value));raise SystemExit(0 if value['passed'] and not value['CUDA_initialized'] else 1)
