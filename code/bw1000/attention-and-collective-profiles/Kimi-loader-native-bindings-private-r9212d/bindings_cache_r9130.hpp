#pragma once
#include <optional>
#include <stdexcept>
#include <thread>
#include <cstdint>
#include <exception>
namespace loading_r9130 {
// One cache per explicit owner/model loading session, not process-wide or pointer-only TLS.
// Binding owns Weight/Tensor handles. Close/abort on owner thread BEFORE Runtime teardown.
template<class Bindings> class ImmutableBindings {
 const void*model_;std::uint64_t generation_;std::thread::id owner_;bool active_=true;
 std::optional<Bindings> bindings_;
 void check(const void*m,std::uint64_t gen)const{if(std::this_thread::get_id()!=owner_||m!=model_||gen!=generation_||!active_)throw std::logic_error("loading cache owner/model/generation/state mismatch");}
public:
 std::uint64_t builds=0,hits=0,initial_keys=0;
 ImmutableBindings(const void*m,std::uint64_t gen):model_(m),generation_(gen),owner_(std::this_thread::get_id()){if(!m||!gen)throw std::invalid_argument("loading identity");}
 ImmutableBindings(const ImmutableBindings&)=delete;ImmutableBindings&operator=(const ImmutableBindings&)=delete;
 ~ImmutableBindings()noexcept{if(active_&&bindings_)std::terminate();}
 template<class Build>const Bindings&get(const void*m,std::uint64_t gen,Build build){check(m,gen);if(!bindings_){bindings_.emplace(build());++builds;initial_keys=bindings_->size();}else ++hits;return *bindings_;}
 template<class Build,class Equal>void finish(const void*m,std::uint64_t gen,Build build,Equal equal){check(m,gen);try{auto fresh=build();++builds;if(bindings_&&!equal(*bindings_,fresh))throw std::logic_error("late registration/lazy parameter/storage/shape/dtype changed during load");bindings_.reset();active_=false;}catch(...){bindings_.reset();active_=false;throw;}}
 void abort(const void*m,std::uint64_t gen){check(m,gen);bindings_.reset();active_=false;}
 bool holds_bindings()const noexcept{return bindings_.has_value();}
};
}
