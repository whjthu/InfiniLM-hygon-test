import pathlib,ast
r=pathlib.Path.cwd();s=(r/'modern_Deep_scratch_Model_r8956.py').read_text().replace('reusedDSO8936','reusedDSO8954');(r/'modern_Deep_scratch_Model_r8956.py').write_bytes(s.encode())
s=(r/'check_Deep_latent_Model_consumer_r8943.py').read_text().replace('Deep_latent_Model','Deep_scratch_Model').replace('Deep-latent-Model','Deep-scratch-Model').replace('r8943','r8956').replace('latent','scratch');s=s.replace('self.deep_scratch_prefill_mode_r8933=Fn(self.mode);self.deep_scratch_prefill_counts_r8933=Fn(self.scratch)', 'self.deep_latent_prefill_mode_r8933=Fn(lambda a:0);self.deep_reference_scratch_mode_r8953=Fn(self.mode);self.deep_reference_scratch_counts_r8954=Fn(self.scratch)')
s=s.replace('self.deep_owner_arm_r8900=Fn(self.armfn);', 'self.modern_deep_latent_fused_calls_r681=Fn(lambda d:self.dec*43);self.deep_owner_arm_r8900=Fn(self.armfn);')
s=s.replace('vals=[self.tiled*43,self.tiled*43*2048,self.tiled*43*64]', 'vals=[(self.pre-self.tiled)*43,self.tiled*43,(self.pre-self.tiled)*43*6144,self.tiled*43*3]')
s=s.replace("vals[2]+=1", "vals[3]+=1")
ast.parse(s);(r/'check_Deep_scratch_Model_consumer_r8956.py').write_bytes(s.encode());print('CPU Model fakeconsumer prepared')
