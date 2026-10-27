 // Included inside the actual ProbeEngine derived from InferEngine.
 using PrefixGroupR128=::infinilm::models::deepseek_v4::prefix_r128::Group;
 using PrefixRowR128=::infinilm::models::deepseek_v4::prefix_r128::Row;
 using PrefixActionR128=::infinilm::models::deepseek_v4::prefix_r128::Action;
 std::shared_ptr<PrefixGroupR128>prefix_group_r128;
 std::vector<PrefixRowR128>prefix_rows_r128;
 bool prefix_poisoned_r128=false,prefix_verified_r128=false;
 size_t prefix_physical_rows_r128=0;uint64_t prefix_generation_r128=1,prefix_last_tx_r128=0;
 void prefix_dispatch_r128(PrefixActionR128 action,const std::vector<size_t>&counts){
  std::array<bool,8>posted{};std::exception_ptr first;
  auto g=prefix_group_r128;auto rows=prefix_rows_r128;
  for(size_t rank=0;rank<8;++rank){
   try{workers_.at(rank)->submit_owner_task_r128([g,rows,counts,rank,action](::infinilm::InfinilmModel&m,rt::Runtime&r){
    try{::infinilm::models::deepseek_v4::prefix_r128::native_control(m,r,rank,action,g,rows,counts);}
    catch(...){g->poison("actual8owner command failure");throw;}
   });posted[rank]=true;}catch(...){if(!first)first=std::current_exception();g->poison("owner submit failure");}
  }
  for(size_t rank=0;rank<8;++rank)if(posted[rank])try{workers_[rank]->wait_owner_task_r128();}catch(...){if(!first)first=std::current_exception();g->poison("owner command failure");}
  if(first){prefix_poisoned_r128=true;std::rethrow_exception(first);}
 }
 void prefix_begin_r128(uint64_t tx,std::vector<PrefixRowR128>rows,size_t physical_rows){
  std::lock_guard<std::mutex>lock(submission_mutex_);
  const char*enabled=std::getenv("INFINILM_DEEP_PREFIX_R128");
  if(!enabled||std::strcmp(enabled,"1")||prefix_poisoned_r128||prefix_group_r128||workers_.size()!=8||weight_state_!=WeightState::Ready||!tx||tx<=prefix_last_tx_r128||rows.empty()||rows.size()>128)throw std::logic_error("prefix defaultOFF/unready/stale/busy");
  prefix_physical_rows_r128=physical_rows;if(!physical_rows||physical_rows>128||native_sparse_lengths_.size()!=physical_rows)throw std::invalid_argument("prefix physical rows");prefix_group_r128=std::make_shared<PrefixGroupR128>(tx,prefix_generation_r128);prefix_rows_r128=std::move(rows);prefix_verified_r128=false;prefix_last_tx_r128=tx;
  prefix_dispatch_r128(PrefixActionR128::Begin,{});
 }
 Output prefix_forward_r128(const Input&input){
  std::lock_guard<std::mutex>lock(submission_mutex_);
  if(prefix_poisoned_r128||!prefix_group_r128||prefix_verified_r128||!input.sample_all_positions)throw std::logic_error("prefix verification state");
  try{
   // All eight original rank workers perform real model forward+sampling;
   // bypass only InferEngine's ordinary eager length publication.
   for(auto&w:workers_)w->run(input);
   std::exception_ptr first;for(auto&w:workers_)try{w->wait();}catch(...){if(!first)first=std::current_exception();}
   if(first)std::rethrow_exception(first);
   prefix_verified_r128=true;return workers_[0]->get_output();
  }catch(...){prefix_poisoned_r128=true;prefix_group_r128->poison("native forward failed");throw;}
 }
 void prefix_commit_r128(const std::vector<size_t>&counts){
  std::lock_guard<std::mutex>lock(submission_mutex_);
  if(prefix_poisoned_r128||!prefix_group_r128||!prefix_verified_r128||counts.size()!=prefix_rows_r128.size())throw std::logic_error("prefix commit state");
  for(auto count:counts)if(count<1||count>2)throw std::invalid_argument("prefix accepted INPUT prefix must1/2");
  prefix_dispatch_r128(PrefixActionR128::SealPublish,counts);
  if(!prefix_group_r128->complete())throw std::logic_error("prefix missing all8 commit");
  // Constructor/reset preallocates the row vector; no allocation after publish.
  for(size_t i=0;i<counts.size();++i)native_sparse_lengths_.at(prefix_rows_r128[i].row)=prefix_rows_r128[i].past+counts[i];
  prefix_group_r128.reset();prefix_rows_r128.clear();prefix_verified_r128=false;
 }
 void prefix_cancel_r128(){
  std::lock_guard<std::mutex>lock(submission_mutex_);
  if(prefix_poisoned_r128||!prefix_group_r128)throw std::logic_error("prefix cancel state");
  prefix_group_r128->start_cancel();prefix_dispatch_r128(PrefixActionR128::Cancel,{});
  if(!prefix_group_r128->fully_cancelled())throw std::logic_error("prefix missing all8 cancel");
  prefix_group_r128.reset();prefix_rows_r128.clear();prefix_verified_r128=false;
 }
 void prefix_reset_completed_r128(){
  std::lock_guard<std::mutex>lock(submission_mutex_);
  prefix_group_r128.reset();prefix_rows_r128.clear();prefix_verified_r128=false;prefix_poisoned_r128=false;++prefix_generation_r128;
 }
