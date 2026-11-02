#pragma once
#include <cstdint>
#include <deque>
#include <vector>
namespace infinilm::models::glm_moe_dsa::v_on_demand_r403 {
// Work's destructor drains its stream before these member buffers destruct.
// Copying logical/mapped IDs costs HOST time inside the real forward wall.
struct HeldHostIds {
 std::deque<std::vector<int64_t>>values;
 const std::vector<int64_t>&hold(const std::vector<int64_t>&v){values.push_back(v);return values.back();}
};
}
