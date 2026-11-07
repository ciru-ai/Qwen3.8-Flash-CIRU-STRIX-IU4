#include "../node_context.h"
#include "../op_table.h"
#include "../utils.h"

#include <openvino/op/add.hpp>
#include <openvino/op/broadcast.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/gather.hpp>
#include <openvino/op/greater_eq.hpp>
#include <openvino/op/less.hpp>
#include <openvino/op/logical_or.hpp>
#include <openvino/op/range.hpp>
#include <openvino/op/select.hpp>
#include <openvino/op/shape_of.hpp>
#include <openvino/op/squeeze.hpp>
#include <openvino/op/unsqueeze.hpp>

namespace ov {
namespace frontend {
namespace ggml {
namespace op {

// GGML FILL sets all elements of a tensor to a constant value.
// The constant is stored as a float in op_params[0].
OutputVector translate_fill(const NodeContext & context) {
    num_inputs_check(context, 1, 1);

    float c;
    memcpy(&c, context.get_output_op_params(), sizeof(float));

    auto input = context.get_input(0);
    auto val = ov::op::v0::Constant::create(input.get_element_type(), {}, {c});
    auto shape = std::make_shared<ov::op::v3::ShapeOf>(input, ov::element::i64);

    if (context.get_op_case() == 1 && context.has_input("cache_rs_reset_len")) {
        auto idx = context.get_input("cache_rs_reset_idx");
        auto len = context.get_input("cache_rs_reset_len");
        auto zero = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {0});
        auto one = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {1});
        auto slot_dim = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {2});
        auto n_slots = std::make_shared<ov::op::v0::Squeeze>(
            std::make_shared<ov::op::v8::Gather>(shape, slot_dim, zero));
        auto slots = std::make_shared<ov::op::v4::Range>(zero, n_slots, one, ov::element::i64);
        auto end = std::make_shared<ov::op::v1::Add>(idx, len);
        auto keep = std::make_shared<ov::op::v1::LogicalOr>(
            std::make_shared<ov::op::v1::Less>(slots, idx),
            std::make_shared<ov::op::v1::GreaterEqual>(slots, end));
        auto mask = std::make_shared<ov::op::v0::Unsqueeze>(
            keep, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {1}));
        auto res = std::make_shared<ov::op::v1::Select>(mask, input, val);
        return rename_outputs_with_suffix({res}, context.get_name());
    }

    auto res = std::make_shared<ov::op::v3::Broadcast>(val, shape);

    return rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
