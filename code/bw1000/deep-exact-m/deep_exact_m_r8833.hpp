#pragma once
#include "runtime/runtime.hpp"
#include <memory>
#include <string>
namespace infinilm::models::deepseek_v4::bank_exact_m_r8833 {
struct Owner;
struct Lease;
std::shared_ptr<Owner> prepare(runtime::Runtime &);
std::shared_ptr<Lease> make_lease(runtime::Runtime &,const std::shared_ptr<Owner>&);
void finish_report(const std::shared_ptr<Lease>&);
void clear_report();
const std::string &report();
}
