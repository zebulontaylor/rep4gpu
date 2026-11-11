#pragma once

#include "emp-tool/emp-tool.h"
#include "vole/base_svole.h"
#include "vole/base_cot.h"
#include "vole/spfss_sender.h"
#include "vole/spfss_recver.h"
#include "vole/fields/fp61x2.h"
#include <openssl/sha.h>
#include <array>
#include <chrono>
#include <ctime>
#include <exception>
#include <limits>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace matrix_vope {
using Digest = std::array<unsigned char, SHA256_DIGEST_LENGTH>;

inline Digest digest(const void* data, size_t size) {
    Digest out;
    SHA256(static_cast<const unsigned char*>(data), size, out.data());
    return out;
}

inline void require_secure_randomness() {
    const char* mode = std::getenv("EMP_TEST_MODE");
    if ((mode && mode[0] == '1') || emp::is_test_mode())
        throw std::runtime_error("EMP deterministic test mode is not allowed");
}

template<typename Field = FP61>
inline Field uniform(emp::PRG& prg) {
    decltype(Field{}.val) value;
    do {
        prg.random_data_unaligned(&value, sizeof(value));
        value &= Field::PR_mask;
    } while (value == Field::PR);
    return Field(value, false);
}

template<typename Field>
inline std::vector<unsigned char> encode_fields(const Field* values, size_t count) {
    std::vector<unsigned char> out(count * Field::DS_byte_len);
    for (size_t i = 0; i < count; ++i) {
        if (values[i].val >= Field::PR) throw std::runtime_error("noncanonical field value");
        for (size_t byte = 0; byte < Field::DS_byte_len; ++byte)
            out[i * Field::DS_byte_len + byte] = values[i].val >> (8 * byte);
    }
    return out;
}

template<typename Field, typename IO>
inline void send_fields(IO& io, const Field* values, size_t count) {
    auto bytes = encode_fields(values, count);
    io.send_data(bytes.data(), bytes.size());
}

template<typename Field, typename IO>
inline void receive_fields(IO& io, Field* values, size_t count) {
    std::vector<unsigned char> bytes(count * Field::DS_byte_len);
    io.recv_data(bytes.data(), bytes.size());
    for (size_t i = 0; i < count; ++i) {
        decltype(Field{}.val) value = 0;
        for (size_t byte = 0; byte < Field::DS_byte_len; ++byte)
            value |= decltype(value)(bytes[i * Field::DS_byte_len + byte]) << (8 * byte);
        if (value >= Field::PR) throw std::runtime_error("noncanonical field encoding");
        values[i] = Field(value, false);
    }
}

// This bare COPE path is not maliciously private: selective acceptance can
// reveal a scalar-key bit. Its polynomial check is not a malicious upgrade.
// The COPE bit decomposition is unchanged. Wide fields use typed canonical
// buffers because the backend's bulk IO bypasses the field's decoder.
template<typename Field, typename IO>
inline void canonical_cope(IO& io, int party, CopeFp<IO, Field>& cope,
                           Field* output, Field* values, size_t count, const std::string& fault = "") {
    std::fill(output, output + count, Field{});
    std::vector<Field> correction(count);
    Field power(1);
    for (size_t bit = 0; bit < Field::slot_stride_bits; ++bit) {
        if (party == emp::BOB) {
            std::vector<Field> first(count), second(count);
            for (size_t row = 0; row < count; ++row) {
                first[row] = uniform<Field>(cope.G0[bit]);
                second[row] = uniform<Field>(cope.G1[bit]);
                correction[row] = first[row] - second[row] - values[row];
                output[row] = output[row] + first[row] * power;
            }
            if (fault == "cope-high" && bit == (Field::PR_bit_len > 64 ? 64 : Field::PR_bit_len - 1))
                correction[0] = correction[0] + Field(1);
            send_fields(io, correction.data(), count);
        } else {
            std::vector<Field> selected(count);
            for (auto& value : selected) value = uniform<Field>(cope.G0[bit]);
            receive_fields(io, correction.data(), count);
            for (size_t row = 0; row < count; ++row) {
                if (cope.delta_bool[bit]) selected[row] = selected[row] + correction[row];
                output[row] = output[row] + selected[row] * power;
            }
        }
        power = power + power;
    }
    io.flush();
}

// One private entropy root per endpoint. Every substream consumes a new master
// block and mixes its purpose and advancing index before generating output.
class EndpointEntropy {
    static emp::PRG secure_root() {
        require_secure_randomness();
        return emp::PRG{};
    }
    emp::PRG master_;
    uint64_t next_ = 0;
    bool retired_ = false;
public:
    EndpointEntropy() : master_(secure_root()) {}
    void ensure_live() const {
        require_secure_randomness();
        if (retired_) throw std::runtime_error("entropy context retired after abort");
    }
    void retire() noexcept { retired_ = true; }
    uint64_t substreams() const { return next_; }
    emp::PRG fork(const std::string& purpose) {
        ensure_live();
        if (next_ == std::numeric_limits<uint64_t>::max()) {
            retire();
            throw std::runtime_error("entropy substream counter exhausted");
        }
        emp::block seed, separation;
        master_.random_block(&seed, 1);
        auto purpose_hash = digest(purpose.data(), purpose.size());
        std::memcpy(&separation, purpose_hash.data(), sizeof(separation));
        seed ^= separation ^ emp::makeBlock(0, next_++);
        return emp::PRG(&seed);
    }
};

struct RetireEntropyOnFailure {
    EndpointEntropy& source;
    int exceptions = std::uncaught_exceptions();
    ~RetireEntropyOnFailure() {
        if (std::uncaught_exceptions() > exceptions) source.retire();
    }
};

// Both commitments are fixed before either opening is delivered.
template<typename IO>
inline std::vector<unsigned char> commit_exchange(IO& io, int party, EndpointEntropy& entropy,
                                                 const void* data, size_t size,
                                                 const std::string& domain) {
    if (party != emp::ALICE && party != emp::BOB)
        throw std::invalid_argument("invalid commitment sender role");
    RetireEntropyOnFailure retire{entropy};
    auto prg = entropy.fork(domain + "/commitment-nonce/role-" + std::to_string(party));
    std::array<unsigned char, 32> nonce;
    prg.random_data_unaligned(nonce.data(), nonce.size());
    std::vector<unsigned char> transcript(domain.begin(), domain.end());
    transcript.push_back(static_cast<unsigned char>(party));
    transcript.insert(transcript.end(), nonce.begin(), nonce.end());
    const auto* bytes = static_cast<const unsigned char*>(data);
    transcript.insert(transcript.end(), bytes, bytes + size);
    auto commitment = digest(transcript.data(), transcript.size());
    Digest peer_commitment;
    io.send_data(commitment.data(), commitment.size());
    io.flush();
    io.recv_data(peer_commitment.data(), peer_commitment.size());
    io.send_data(nonce.data(), nonce.size());
    io.send_data(data, size);
    io.flush();
    std::array<unsigned char, 32> peer_nonce;
    std::vector<unsigned char> peer(size);
    io.recv_data(peer_nonce.data(), peer_nonce.size());
    io.recv_data(peer.data(), peer.size());
    transcript.assign(domain.begin(), domain.end());
    transcript.push_back(static_cast<unsigned char>(3 - party));
    transcript.insert(transcript.end(), peer_nonce.begin(), peer_nonce.end());
    transcript.insert(transcript.end(), peer.begin(), peer.end());
    if (digest(transcript.data(), transcript.size()) != peer_commitment)
        throw std::runtime_error("commitment opening failed");
    return peer;
}

template<typename IO>
inline emp::block joint_coin(IO& io, int party, EndpointEntropy& entropy,
                             const std::string& domain) {
    RetireEntropyOnFailure retire{entropy};
    auto prg = entropy.fork(domain + "/coin-value/role-" + std::to_string(party));
    std::array<unsigned char, 16> mine;
    prg.random_data_unaligned(mine.data(), mine.size());
    auto peer = commit_exchange(io, party, entropy, mine.data(), mine.size(), domain);
    for (size_t i = 0; i < mine.size(); ++i) mine[i] ^= peer[i];
    emp::block seed;
    std::memcpy(&seed, mine.data(), sizeof(seed));
    return seed;
}


struct Config {
    size_t output_length = 128;
    size_t weight = 8;
    size_t log_block = 6;
    size_t checks = 3;
    size_t leaves() const { return size_t(1) << log_block; }
    size_t noise_length() const { return weight * leaves(); }
    void validate() const {
        if (!output_length || output_length > 4096 || !weight || weight > 256 ||
            log_block < 1 || log_block > 12 || checks < 1 || checks > 16 ||
            noise_length() > 65536)
            throw std::invalid_argument("unsupported experimental configuration");
    }
};

// The two extension engines have opposite OT directions. Their base OTs run
// once per peer context; each scalar key consumes fresh chosen ROT indices.
template<typename Field, typename IO = emp::NetIO>
struct PeerContext {
    ProgBaseCot<IO> tree_ot;
    emp::SoftSpoken<4> cope_ot;
    std::vector<emp::PRG> cope_prg0, cope_prg1;
    uint64_t cope_indices = 0;
    uint64_t tree_indices = 0;
    PeerContext(IO& io, int party)
        : tree_ot((require_secure_randomness(), party), &io, true), cope_ot(3 - party, &io, true) {
        tree_ot.ext->set_sid(emp::makeBlock(0x4d56545245454f54ULL, 1));
        cope_ot.set_sid(emp::makeBlock(0x4d56434f50454f54ULL, 1));
        tree_ot.cot_gen_pre();
        // These initial AES schedules never generate output. A fresh ROT key
        // replaces each schedule before every COPE call, avoiding discarded
        // entropy draws from default-constructing a new PRG array each time.
        cope_prg0.reserve(Field::slot_stride_bits);
        if (party == emp::BOB) cope_prg1.reserve(Field::slot_stride_bits);
        for (size_t bit = 0; bit < Field::slot_stride_bits; ++bit) {
            cope_prg0.emplace_back(&emp::zero_block);
            if (party == emp::BOB) cope_prg1.emplace_back(&emp::zero_block);
        }
    }
};

template<typename Field, typename Packed>
inline void checked_base_extended(emp::NetIO& io, int party, Field delta,
                                  size_t count, std::vector<Field>& keys,
                                  std::vector<Packed>& macs,
                                  std::vector<Field>& values,
                                  PeerContext<Field>& peer, EndpointEntropy& entropy_source, const std::string& domain,
                                  std::array<double, 4>& wall,
                                  std::array<double, 4>& cpu, const std::string& fault = "") {
    RetireEntropyOnFailure retire{entropy_source};
    entropy_source.ensure_live();
    auto wall_start = std::chrono::steady_clock::now();
    auto cpu_start = std::clock();
    auto mark = [&](size_t phase) {
        auto wall_now = std::chrono::steady_clock::now();
        auto cpu_now = std::clock();
        wall[phase] = std::chrono::duration<double>(wall_now - wall_start).count();
        cpu[phase] = double(cpu_now - cpu_start) / CLOCKS_PER_SEC;
        wall_start = wall_now;
        cpu_start = cpu_now;
    };
    constexpr size_t bits = Field::slot_stride_bits;
    CopeFp<emp::NetIO, Field> cope(party, &io);
    struct BorrowedPrgs {
        CopeFp<emp::NetIO, Field>& cope;
        ~BorrowedPrgs() { cope.G0 = nullptr; cope.G1 = nullptr; }
    } borrowed{cope};
    std::vector<emp::block> first(bits), second(bits);
    if (party == emp::ALICE) {
        cope.delta = delta;
        cope.delta_bool = new bool[bits];
        cope.delta64_to_bool(cope.delta_bool, delta.val, bits);
        peer.cope_ot.recv_rot(first.data(), cope.delta_bool, bits);
    } else {
        peer.cope_ot.send_rot(first.data(), second.data(), bits);
    }
    io.flush();
    mark(0);
    cope.G0 = peer.cope_prg0.data();
    if (party == emp::BOB) cope.G1 = peer.cope_prg1.data();
    for (size_t bit = 0; bit < bits; ++bit) {
        cope.G0[bit].reseed(&first[bit]);
        if (party == emp::BOB) cope.G1[bit].reseed(&second[bit]);
    }
    mark(1);
    io.flush();
    peer.cope_indices += bits;
    std::vector<Field> correlation(count + 1);
    std::vector<Field> chi(count);
    if (party == emp::ALICE) {
        if constexpr (Field::PR_bit_len > 61)
            canonical_cope(io, party, cope, correlation.data(), static_cast<Field*>(nullptr), count + 1);
        else cope.extend64(correlation.data(), count + 1, bits, 1);
        mark(2);
        auto entropy = entropy_source.fork(domain + "/base-check-seed");
        auto seed = uniform<Field>(entropy);
        seed.send(&io);
        io.flush();
        check_coeff_gen(chi.data(), seed, count);
        Field expected = field_inn_prdt_sum_red(correlation.data(), chi.data(), count)
                        + correlation[count];
        Field response[2];
        receive_fields(io, response, 2);
        const bool valid = response[0].val < Field::PR && response[1].val < Field::PR &&
            expected + delta * response[1] == response[0];
        unsigned char accepted = valid ? 1 : 0;
        io.send_data(&accepted, 1);
        io.flush();
        if (!valid) throw std::runtime_error("extended-OT base VOLE consistency check failed");
        std::copy(correlation.begin(), correlation.begin() + count, keys.begin());
    } else {
        if constexpr (Field::PR_bit_len > 61)
            canonical_cope(io, party, cope, correlation.data(), values.data(), count + 1, fault);
        else cope.extend64(correlation.data(), values.data(), count + 1, bits, 1);
        io.flush();
        mark(2);
        if (fault == "base-mac") correlation[0] = correlation[0] + Field(1);
        Field seed;
        seed.recv(&io);
        if (seed.val >= Field::PR) throw std::runtime_error("noncanonical base challenge");
        check_coeff_gen(chi.data(), seed, count);
        Field response[2] = {
            field_inn_prdt_sum_red(correlation.data(), chi.data(), count) + correlation[count],
            field_inn_prdt_sum_red(values.data(), chi.data(), count) + values[count]};
        if (fault == "base-response0") response[0] = response[0] + Field(1);
        if (fault == "base-response1") response[1] = response[1] + Field(1);
        send_fields(io, response, 2);
        io.flush();
        unsigned char accepted;
        io.recv_data(&accepted, 1);
        if (accepted != 1) throw std::runtime_error("extended-OT base VOLE consistency check failed");
        for (size_t i = 0; i < count; ++i) macs[i].setLowHigh(correlation[i], values[i]);
    }
    mark(3);
}

template<typename Field>
struct PublicMapCache {
    size_t rows = 0, columns = 0;
    std::vector<Field> coefficients;
};

template<typename Field>
struct Output {
    std::array<double, 7> phase_seconds{};
    std::array<double, 4> base_wall_seconds{}, base_cpu_seconds{};
    std::vector<Field> value;  // Only the programmer receives values.
    std::vector<Field> share; // MAC at programmer, key at scalar holder.
};

// Le Mans chosen single-point VOLE followed by its public dual-LPN map.
// ALICE holds Delta. BOB holds a private regular-noise seed. All base VOLE,
// chosen OT, and single-point checks execute on every independent call.
template<typename Field, typename Packed>
inline Output<Field> seeded_vole(emp::NetIO& io, int party, Field delta,
                          emp::block noise_seed, const Config& cfg,
                          size_t session, const std::string& connection_domain, EndpointEntropy& entropy_source,
                          const std::string& cheat = "",
                          PeerContext<Field>* persistent = nullptr, PublicMapCache<Field>* public_map = nullptr) {
    RetireEntropyOnFailure retire{entropy_source};
    entropy_source.ensure_live();
    using Clock = std::chrono::steady_clock;
    auto phase_start = Clock::now();
    auto mark = [&]() {
        auto now = Clock::now();
        double seconds = std::chrono::duration<double>(now - phase_start).count();
        phase_start = now;
        return seconds;
    };
    std::array<double, 7> phases{};
    std::array<double, 4> base_wall{}, base_cpu{};
    cfg.validate();
    std::array<uint64_t, 8> configuration{2, Field::PR_bit_len, cfg.output_length, cfg.weight,
                                         cfg.log_block, cfg.checks, session, persistent ? 1UL : 0UL};
    auto peer_config = commit_exchange(io, party, entropy_source, configuration.data(), sizeof(configuration),
                                      connection_domain + "/seeded-config-v1");
    if (std::memcmp(peer_config.data(), configuration.data(), sizeof(configuration)))
        throw std::runtime_error("public configuration mismatch");
    phases[0] = mark();
    size_t leaves = cfg.leaves(), count = cfg.weight * (1 + cfg.checks);
    const std::string sid = connection_domain + "/seeded-v1/" + std::to_string(session);
    std::vector<Field> base_keys(count), base_values(count + 1);
    std::vector<Packed> base_macs(count);
    auto entropy = entropy_source.fork(sid + "/base-values-and-ggm-roots");
    if (party == emp::BOB)
        for (auto& value : base_values) value = uniform<Field>(entropy);
    std::unique_ptr<PeerContext<Field>> fresh_wide_context;
    if constexpr (Field::PR_bit_len > 61) {
        if (!persistent) fresh_wide_context = std::make_unique<PeerContext<Field>>(io, party);
    }
    auto* base_peer = persistent ? persistent : fresh_wide_context.get();
    if (base_peer) {
        checked_base_extended<Field, Packed>(io, party, delta, count, base_keys, base_macs, base_values, *base_peer, entropy_source, sid, base_wall, base_cpu, cheat);
    } else if (party == emp::ALICE) {
        BaseSvoleFp<emp::NetIO, Field> base(party, &io, delta);
        base.compute_send64(base_keys.data(), count);
        io.flush();
    } else {
        BaseSvoleFp<emp::NetIO, Field> base(party, &io);
        base.compute_recv64(base_macs.data(), base_values.data(), count);
        io.flush();
    }

    phases[1] = mark();
    emp::PRG noise(&noise_seed);
    std::vector<size_t> positions(cfg.weight);
    std::vector<Field> betas(cfg.weight);
    std::unique_ptr<bool[]> choices(new bool[cfg.weight * cfg.log_block]);
    if (party == emp::BOB) {
        for (size_t i = 0; i < cfg.weight; ++i) {
            uint64_t random_index;
            noise.random_data_unaligned(&random_index, sizeof(random_index));
            positions[i] = random_index & (leaves - 1);
            do { betas[i] = uniform<Field>(noise); } while (betas[i] == Field{});
            for (size_t bit = 0; bit < cfg.log_block; ++bit)
                choices[i * cfg.log_block + bit] =
                    !((positions[i] >> (cfg.log_block - bit - 1)) & 1);
        }
    }
    emp::OTPre<emp::NetIO> pre(&io, cfg.log_block, cfg.weight);
    std::unique_ptr<ProgBaseCot<emp::NetIO>> fresh_ot;
    if (!base_peer) {
        fresh_ot = std::make_unique<ProgBaseCot<emp::NetIO>>(party, &io, true);
        fresh_ot->cot_gen_pre();
    }
    auto& cot = base_peer ? base_peer->tree_ot : *fresh_ot;
    if (party == emp::ALICE) cot.prog_cot_gen(&pre, cfg.weight * cfg.log_block);
    else cot.prog_cot_gen(&pre, choices.get(), cfg.weight * cfg.log_block);
    if (persistent) persistent->tree_indices += cfg.weight * cfg.log_block;

    phases[2] = mark();
    std::vector<Field> sparse_values(cfg.noise_length()), sparse_shares(cfg.noise_length());
    for (size_t tree = 0; tree < cfg.weight; ++tree) {
        size_t offset = tree * (1 + cfg.checks);
        std::vector<Field> current(leaves);
        if (party == emp::ALICE) {
            Field correction;
            correction.recv(&io);
            if (correction.val >= Field::PR) throw std::runtime_error("noncanonical field correction");
            Field key_beta = base_keys[offset] - delta * correction;
            emp::block root;
            entropy.random_block(&root, 1);
            SpfssSenderFp<emp::NetIO, Field> sender(&io, cfg.log_block + 1, root);
            sender.compute(current.data(), delta, key_beta);
            if (cheat == "tree" && tree == 0) sender.secret_sum = sender.secret_sum + Field(1);
            sender.send(&pre, &io, tree);
        } else {
            Field correction = betas[tree] - base_macs[offset].getHigh();
            correction.send(&io);
            io.flush();
            SpfssRecverFp<emp::NetIO, Field> receiver(&io, cfg.log_block + 1);
            std::memcpy(receiver.b, choices.get() + tree * cfg.log_block,
                        cfg.log_block * sizeof(bool));
            if (receiver.get_index() != positions[tree])
                throw std::runtime_error("punctured index mapping failed");
            receiver.recv(&pre, &io, tree);
            if (receiver.share.val >= Field::PR) throw std::runtime_error("noncanonical GGM correction");
            std::vector<Packed> tags(leaves);
            receiver.compute(tags.data(), base_macs[offset]);
            for (size_t j = 0; j < leaves; ++j) current[j] = tags[j].getLow();
            if ((cheat == "mac" || cheat == "offset") && tree == 0)
                current[0] = current[0] + Field(1);
            if (cheat == "offset" && tree == 1) current[0] = current[0] - Field(1);
            sparse_values[tree * leaves + positions[tree]] = betas[tree];
        }
        std::copy(current.begin(), current.end(), sparse_shares.begin() + tree * leaves);
    }
    phases[3] = mark();
    // Every tree correction is fixed before this common challenge. Each point
    // retains its independent base mask and independent leaf coefficients.
    for (size_t repeat = 0; repeat < cfg.checks; ++repeat) {
        const auto domain = sid + "/batch-check-" + std::to_string(repeat);
        auto coin = joint_coin(io, party, entropy_source, domain + "/coin");
        emp::PRG challenge(&coin);
        std::vector<Field> residuals(cfg.weight), corrections(cfg.weight);
        std::vector<Field> selected(cfg.weight);
        for (size_t tree = 0; tree < cfg.weight; ++tree) {
            for (size_t j = 0; j < leaves; ++j) {
                auto chi = uniform<Field>(challenge);
                residuals[tree] = residuals[tree] + chi * sparse_shares[tree * leaves + j];
                if (party == emp::BOB && j == positions[tree]) selected[tree] = chi;
            }
            const size_t mask = tree * (1 + cfg.checks) + 1 + repeat;
            if (party == emp::BOB) {
                corrections[tree] = betas[tree] * selected[tree] - base_macs[mask].getHigh();
                residuals[tree] = residuals[tree] - base_macs[mask].getLow();
            }
        }
        if (party == emp::BOB) {
            if (cheat == "correction" && repeat == 0) corrections[0] = corrections[0] + Field(1);
            send_fields(io, corrections.data(), corrections.size());
            io.flush();
        } else {
            receive_fields(io, corrections.data(), corrections.size());
            for (size_t tree = 0; tree < cfg.weight; ++tree) {
                if (corrections[tree].val >= Field::PR)
                    throw std::runtime_error("noncanonical check correction");
                const size_t mask = tree * (1 + cfg.checks) + 1 + repeat;
                residuals[tree] = residuals[tree] + delta * corrections[tree] - base_keys[mask];
            }
        }
        // Compare the entire vector. No cancellation between point residuals
        // is possible except through breaking the hash comparison.
        auto residual_wire = encode_fields(residuals.data(), residuals.size());
        auto mine = digest(residual_wire.data(), residual_wire.size());
        auto other = commit_exchange(io, party, entropy_source, mine.data(), mine.size(), domain + "/vector-equality");
        if (!std::equal(mine.begin(), mine.end(), other.begin()))
            throw std::runtime_error("single-point VOLE consistency check failed");
    }

    phases[4] = mark();
    // The small dense H is public and fixed across scalar keys. Its parameters
    // are for functionality experiments; no LPN security level is assigned.
    PublicMapCache<Field> local_map;
    auto& map = public_map ? *public_map : local_map;
    if (map.rows != cfg.output_length || map.columns != cfg.noise_length()) {
        map.rows = cfg.output_length;
        map.columns = cfg.noise_length();
        emp::block code_seed = emp::makeBlock(0x4d4154524958564fULL, 0x50454c454d414e53ULL);
        emp::PRG code(&code_seed);
        map.coefficients.resize(map.rows * map.columns);
        for (auto& coefficient : map.coefficients) coefficient = uniform<Field>(code);
    }
    phases[5] = mark();
    Output<Field> out;
    out.share.resize(cfg.output_length);
    if (party == emp::BOB) out.value.resize(cfg.output_length);
    for (size_t row = 0; row < cfg.output_length; ++row) {
        for (size_t col = 0; col < cfg.noise_length(); ++col) {
            auto coefficient = map.coefficients[row * cfg.noise_length() + col];
            out.share[row] = out.share[row] + coefficient * sparse_shares[col];
            if (party == emp::BOB)
                out.value[row] = out.value[row] + coefficient * sparse_values[col];
        }
    }
    phases[6] = mark();
    out.phase_seconds = phases;
    out.base_wall_seconds = base_wall;
    out.base_cpu_seconds = base_cpu;
    return out;
}
} // namespace matrix_vope
