#pragma once
#include "seeded_vole.hpp"

namespace matrix_vope {

struct SeedTree {
    size_t leaves;
    std::vector<emp::block> seeds, parity0, parity1;
    explicit SeedTree(size_t count) : leaves(count) {}
    void expand(emp::block root) {
        std::vector<emp::block> level(1, root);
        TwoKeyPRP prp(emp::zero_block, emp::makeBlock(0, 1));
        while (level.size() < leaves) {
            std::vector<emp::block> next(level.size() * 2);
            emp::block even = emp::zero_block, odd = emp::zero_block;
            for (size_t i = 0; i < level.size(); ++i) {
                prp.node_expand_1to2(next.data() + 2 * i, level[i]);
                even ^= next[2 * i];
                odd ^= next[2 * i + 1];
            }
            parity0.push_back(even);
            parity1.push_back(odd);
            level.swap(next);
        }
        seeds.swap(level);
    }
    void reconstruct(size_t alpha, const std::vector<emp::block>& keys) {
        std::vector<emp::block> level(1, emp::zero_block);
        TwoKeyPRP prp(emp::zero_block, emp::makeBlock(0, 1));
        size_t missing = 0, depth = keys.size();
        for (size_t layer = 0; layer < depth; ++layer) {
            std::vector<emp::block> next(level.size() * 2, emp::zero_block);
            for (size_t i = 0; i < level.size(); ++i)
                if (i != missing) prp.node_expand_1to2(next.data() + 2 * i, level[i]);
            size_t bit = (alpha >> (depth - layer - 1)) & 1;
            size_t sibling = 2 * missing + (1 - bit);
            emp::block sum = emp::zero_block;
            for (size_t i = 1 - bit; i < next.size(); i += 2) sum ^= next[i];
            next[sibling] = keys[layer] ^ sum;
            missing = 2 * missing + bit;
            level.swap(next);
        }
        seeds.swap(level);
    }
};

template<typename Field>
struct SpvopeContext {
    PeerContext<Field> inner;
    ProgBaseCot<emp::NetIO> vector_tree_ot;
    PublicMapCache<Field> map;
    SpvopeContext(emp::NetIO& io, int party)
        : inner(io, party), vector_tree_ot(3 - party, &io, true) {
        vector_tree_ot.ext->set_sid(emp::makeBlock(0x4d56564543544f54ULL, 1));
        vector_tree_ot.cot_gen_pre();
    }
};

template<typename Field>
struct SpvopeOutput {
    std::vector<Field> matrix_share;
    std::vector<Field> right_vector;
    Field first_delta{}, second_delta{};
};

// This computes the private seeded vector locally, without communication. It
// supports the explicit seed-equivocation diagnostic; it is not a commitment.
template<typename Field>
inline std::vector<Field> local_right_vector(emp::block seed, const Config& cfg,
                                             const PublicMapCache<Field>& map,
                                             bool direct) {
    emp::PRG noise(&seed);
    std::vector<Field> value(cfg.output_length);
    if (direct) {
        for (auto& entry : value) entry = uniform<Field>(noise);
        return value;
    }
    for (size_t tree = 0; tree < cfg.weight; ++tree) {
        uint64_t position;
        noise.random_data_unaligned(&position, sizeof(position));
        position &= cfg.leaves() - 1;
        Field beta;
        do { beta = uniform<Field>(noise); } while (beta == Field{});
        for (size_t row = 0; row < cfg.output_length; ++row)
            value[row] = value[row] + map.coefficients[row * cfg.noise_length() +
                tree * cfg.leaves() + position] * beta;
    }
    return value;
}

template<typename Field, typename Packed>
inline Output<Field> right_mask(emp::NetIO& io, int party, Field delta,
                               emp::block seed, const Config& cfg, size_t session,
                               const std::string& domain, EndpointEntropy& entropy,
                               SpvopeContext<Field>& context, bool direct) {
    if (!direct)
        return seeded_vole<Field, Packed>(io, party, delta, seed, cfg, session,
            domain, entropy, "", &context.inner, &context.map);
    Output<Field> out;
    std::vector<Field> keys(cfg.output_length), values(cfg.output_length + 1);
    std::vector<Packed> tags(cfg.output_length);
    if (party == emp::BOB) {
        auto chosen = local_right_vector<Field>(seed, cfg, context.map, true);
        std::copy(chosen.begin(), chosen.end(), values.begin());
        auto mask_rng = entropy.fork(domain + "/direct-mask");
        values.back() = uniform<Field>(mask_rng);
    }
    std::array<double, 4> wall{}, cpu{};
    checked_base_extended<Field, Packed>(io, party, delta, cfg.output_length,
        keys, tags, values, context.inner, entropy, domain, wall, cpu);
    out.share.resize(cfg.output_length);
    if (party == emp::ALICE) out.share = std::move(keys);
    else {
        out.value.assign(values.begin(), values.begin() + cfg.output_length);
        for (size_t j = 0; j < cfg.output_length; ++j) out.share[j] = tags[j].getLow();
    }
    return out;
}

// Candidate vector-leaf single-point VOPE. Its checked relation targets the
// SECOND call's v. Without same-v binding this is not a malicious primitive
// for arbitrary low-entropy beta; the harness demonstrates selective abort.
template<typename Field, typename Packed>
inline SpvopeOutput<Field> candidate_spvope(emp::NetIO& io, int party, size_t a,
    size_t alpha, Field beta, emp::block right_seed, const Config& cfg,
    size_t session, const std::string& connection_domain, EndpointEntropy& entropy,
    SpvopeContext<Field>& context, bool direct, const std::string& fault = "") {
    RetireEntropyOnFailure retire{entropy};
    entropy.ensure_live();
    cfg.validate();
    if (a < 2 || a > 1024 || (a & (a - 1)) || (party == emp::ALICE && alpha >= a))
        throw std::invalid_argument("invalid single-point VOPE dimensions");
    const size_t b = cfg.output_length;
    const std::string sid = connection_domain + "/spvope/" + std::to_string(session);
    std::array<uint64_t, 8> config{1, Field::PR_bit_len, a, b, cfg.weight,
                                  cfg.log_block, cfg.checks, direct ? 1UL : 0UL};
    auto peer = commit_exchange(io, party, entropy, config.data(), sizeof(config), sid + "/config");
    if (std::memcmp(peer.data(), config.data(), sizeof(config)))
        throw std::runtime_error("single-point VOPE configuration mismatch");
    auto scalar_rng = entropy.fork(sid + "/scalar-masks");
    Field x{}, xstar{};
    if (party == emp::ALICE) x = uniform<Field>(scalar_rng);
    auto first = right_mask<Field, Packed>(io, party, x, right_seed, cfg, 0,
                                           sid + "/first", entropy, context, direct);
    emp::block second_seed = right_seed;
    if (party == emp::BOB && (fault == "seed" || fault == "selective"))
        second_seed ^= emp::makeBlock(0, 1);
    Field correction{};
    if (party == emp::ALICE) { correction = beta - x; send_fields(io, &correction, 1); io.flush(); }
    else receive_fields(io, &correction, 1);

    size_t depth = 0;
    while ((size_t(1) << depth) < a) ++depth;
    std::unique_ptr<bool[]> choices(new bool[depth]);
    if (party == emp::ALICE)
        for (size_t bit = 0; bit < depth; ++bit)
            choices[bit] = !((alpha >> (depth - bit - 1)) & 1);
    emp::OTPre<emp::NetIO> pre(&io, depth, 1);
    if (party == emp::BOB) context.vector_tree_ot.prog_cot_gen(&pre, depth);
    else context.vector_tree_ot.prog_cot_gen(&pre, choices.get(), depth);
    SeedTree tree(a);
    if (party == emp::BOB) {
        auto roots = entropy.fork(sid + "/vector-tree-root");
        emp::block root;
        roots.random_block(&root, 1);
        tree.expand(root);
        pre.send(tree.parity0.data(), tree.parity1.data(), depth, &io, 0);
        io.flush();
    } else {
        std::vector<emp::block> keys(depth);
        pre.recv(keys.data(), choices.get(), depth, &io, 0);
        tree.reconstruct(alpha, keys);
    }
    SpvopeOutput<Field> out;
    out.matrix_share.resize(a * b);
    std::vector<Field> sum(b), d(b);
    for (size_t row = 0; row < a; ++row) {
        if (party == emp::ALICE && row == alpha) continue;
        emp::PRG leaves(&tree.seeds[row]);
        for (size_t col = 0; col < b; ++col) {
            auto value = uniform<Field>(leaves);
            out.matrix_share[row * b + col] = value;
            sum[col] = sum[col] + value;
        }
    }
    if (party == emp::BOB) {
        for (size_t col = 0; col < b; ++col)
            d[col] = first.share[col].negate() - correction * first.value[col] - sum[col];
        if (fault == "tree") d[0] = d[0] + Field(1);
        if (fault == "offset") { d[0] = d[0] + Field(1); if (b > 1) d[1] = d[1] - Field(1); }
        if (fault == "selective") {
            auto second_value = local_right_vector<Field>(second_seed, cfg, context.map, direct);
            for (size_t col = 0; col < b; ++col) d[col] = d[col] + first.value[col] - second_value[col];
        }
        send_fields(io, d.data(), d.size());
        io.flush();
    } else {
        receive_fields(io, d.data(), d.size());
        for (size_t col = 0; col < b; ++col)
            out.matrix_share[alpha * b + col] = first.share[col].negate() - d[col] - sum[col];
        if (fault == "mac") out.matrix_share[0] = out.matrix_share[0] + Field(1);
    }

    // The first tree, OT, and d are fixed before obtaining the fresh check mask.
    if (party == emp::ALICE) xstar = uniform<Field>(scalar_rng);
    auto second = right_mask<Field, Packed>(io, party, xstar, second_seed, cfg, 1,
                                            sid + "/second", entropy, context, direct);
    auto coin = joint_coin(io, party, entropy, sid + "/outer-coin");
    emp::PRG challenges(&coin);
    std::vector<Field> r(a), residual(b);
    for (auto& value : r) value = uniform<Field>(challenges);
    if (party == emp::ALICE) {
        correction = r[alpha] * beta - xstar;
        if (fault == "check") correction = correction + Field(1);
        send_fields(io, &correction, 1);
        io.flush();
    } else receive_fields(io, &correction, 1);
    for (size_t col = 0; col < b; ++col) {
        for (size_t row = 0; row < a; ++row)
            residual[col] = residual[col] + r[row] * out.matrix_share[row * b + col];
        if (party == emp::ALICE) residual[col] = residual[col] + second.share[col];
        else residual[col] = residual[col] + correction * second.value[col] + second.share[col];
    }
    auto wire = encode_fields(residual.data(), residual.size());
    auto mine = digest(wire.data(), wire.size());
    auto other = commit_exchange(io, party, entropy, mine.data(), mine.size(), sid + "/outer-equality");
    if (!std::equal(mine.begin(), mine.end(), other.begin()))
        throw std::runtime_error("outer single-point VOPE consistency check failed");
    if (party == emp::BOB) out.right_vector = std::move(second.value);
    else { out.first_delta = x; out.second_delta = xstar; }
    return out;
}
} // namespace matrix_vope
