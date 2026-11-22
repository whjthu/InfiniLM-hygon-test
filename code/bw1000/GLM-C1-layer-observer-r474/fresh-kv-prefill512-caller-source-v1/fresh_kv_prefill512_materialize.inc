namespace prefill = ::glm_r474_fresh_kv_prefill512;
namespace tf = prefill::layout;
bool prefill_packed = false;
if (prefill::active_for(&r)) {
    const auto bf16 = rt::DataType::kBFloat16;
    const auto prefill_stream = r.stream();
    if (!tf::check(rotary, device_, bf16, tf::rotary)) {
        prefill::record(&r, prefill::rotary_layout);
    } else if (tokens != prefill::tokens || heads_ != prefill::heads || !prefill_stream
               || !tf::check(expanded, device_, bf16, tf::projected)
               || !tf::check(key, device_, bf16, tf::output)) {
        prefill::record(&r, prefill::tensor_layout);
    } else {
        auto candidate_value = work.alloc({tokens, heads_, 256}, bf16);
        if (!tf::check(candidate_value, device_, bf16, tf::output)) {
            prefill::record(&r, prefill::tensor_layout);
        } else if (!tf::disjoint({{{key->data(), tf::bytes(tf::output)},
                                   {candidate_value->data(), tf::bytes(tf::output)},
                                   {expanded->data(), tf::bytes(tf::projected)},
                                   {rotary->data(), tf::bytes(tf::rotary)}}})) {
            prefill::record(&r, prefill::address_span);
        } else {
            if (glm_r474_fresh_kv_pack_t128(key->data(), candidate_value->data(), expanded->data(), rotary->data(),
                                           tokens, heads_, prefill_stream) != 0) {
                throw std::runtime_error("GLM fresh K/V T512 candidate enqueue failed");
            }
            value = std::move(candidate_value);
            prefill::record(&r, prefill::packed);
            prefill_packed = true;
        }
    }
}
