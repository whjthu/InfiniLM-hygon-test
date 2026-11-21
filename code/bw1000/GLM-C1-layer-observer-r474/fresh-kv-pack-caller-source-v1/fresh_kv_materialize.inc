namespace pack = ::glm_r474_fresh_kv_route;
const auto fresh_mode = pack::active_mode(&r);
const auto stream = fresh_mode == pack::Mode::off ? nullptr : r.stream();
rt::Tensor value;
if ((fresh_mode == pack::Mode::tile64 || fresh_mode == pack::Mode::tile128)
    && tokens == 1 && heads_ == 4 && stream
    && pack::matches(expanded, device_, rt::DataType::kBFloat16, pack::projection)
    && pack::matches(rotary, device_, rt::DataType::kBFloat16, pack::rotary_key)
    && pack::matches(key, device_, rt::DataType::kBFloat16, pack::key_or_value)) {
    value = work.alloc({tokens,heads_,256},rt::DataType::kBFloat16);
    if (!pack::matches(value, device_, rt::DataType::kBFloat16, pack::key_or_value)) {
        throw std::runtime_error("GLM fresh K/V allocation differs from its owning Runtime");
    }
    auto launch = fresh_mode == pack::Mode::tile64 ? glm_r474_fresh_kv_pack_t64 : glm_r474_fresh_kv_pack_t128;
    if (launch(key->data(), value->data(), expanded->data(), rotary->data(),
               tokens, heads_, stream) != 0) {
        throw std::runtime_error("GLM fresh K/V pack launch failed");
    }
} else {
    key->narrow({{2,0,192}})->copy_from(r,expanded->narrow({{2,0,192}}));
    for(size_t head=0;head<heads_;++head)key->narrow({{1,head,1},{2,192,64}})->copy_from(r,rotary);
    value=expanded->narrow({{2,192,256}})->contiguous(r);work.held.push_back(value);
}
