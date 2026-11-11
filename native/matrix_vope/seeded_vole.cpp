#include "seeded_vole.hpp"
#include "fp127x2.h"
#include <chrono>
#include <fstream>
#include <iostream>

using namespace matrix_vope;

template<typename Integer>
static std::string decimal(Integer value) {
    std::string out;
    do { out.push_back(char('0' + value % 10)); value /= 10; } while (value);
    std::reverse(out.begin(), out.end());
    return out;
}

template<typename Field>
static void write_fields(std::ostream& out, const std::vector<Field>& values) {
    out << '[';
    for (size_t i = 0; i < values.size(); ++i) {
        if (i) out << ',';
        out << decimal(values[i].val);
    }
    out << ']';
}


template<typename Field, typename Packed>
static int run_field(int argc, char** argv, const char* field_name) {
    try {
        const int party = std::stoi(argv[1]), port = std::stoi(argv[2]);
        const size_t sessions = std::stoul(argv[4]);
        if ((party != emp::ALICE && party != emp::BOB) || !sessions || sessions > 32)
            throw std::invalid_argument("invalid role or session count");
        const std::string cheat = argc > 5 ? argv[5] : "";
        if (!cheat.empty() && cheat != "none" && cheat != "tree" &&
            cheat != "mac" && cheat != "correction" && cheat != "offset" &&
            cheat != "cope-high" && cheat != "base-response0" &&
            cheat != "base-response1" && cheat != "base-mac" && cheat != "force-cope-bit")
            throw std::invalid_argument("unknown fault injection");
        Config cfg;
        if (argc > 6) cfg.output_length = std::stoul(argv[6]);
        if (argc > 7) cfg.weight = std::stoul(argv[7]);
        if (argc > 8) cfg.log_block = std::stoul(argv[8]);
        cfg.validate();
        const std::string mode = argc > 9 ? argv[9] : "persistent";
        if (mode != "persistent" && mode != "fresh") throw std::invalid_argument("unknown OT mode");
        emp::NetIO io(party == emp::ALICE ? nullptr : "127.0.0.1", port);
        const auto context_start = std::chrono::steady_clock::now();
        const auto context_sent_start = io.send_counter;
        const auto context_received_start = io.recv_counter;
        EndpointEntropy entropy_source;
        auto connection_seed = joint_coin(io, party, entropy_source, "matrix-vope/connection-v1");
        auto connection_hash = digest(&connection_seed, sizeof(connection_seed));
        const char* alphabet = "0123456789abcdef";
        std::string connection_domain = "matrix-vope/connection/";
        for (auto byte : connection_hash) {
            connection_domain.push_back(alphabet[byte >> 4]);
            connection_domain.push_back(alphabet[byte & 15]);
        }
        PublicMapCache<Field> public_map;
        std::unique_ptr<PeerContext<Field>> persistent;
        if (mode == "persistent") persistent = std::make_unique<PeerContext<Field>>(io, party);
        auto entropy = entropy_source.fork("matrix-vope/private-inputs/role-" + std::to_string(party));
        const double context_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - context_start).count();
        const auto context_sent_bytes = io.send_counter - context_sent_start;
        const auto context_received_bytes = io.recv_counter - context_received_start;
        emp::block private_seed;
        entropy.random_block(&private_seed, 1);
        std::ofstream output(argv[3]);
        if (!output) throw std::runtime_error("cannot create role output");
        output << "{\"role\":" << party << ",\"security_reviewed\":false,"
               << "\"malicious_privacy\":false,\"field\":\"" << field_name << "\",\"prime_bits\":" << Field::PR_bit_len
               << ",\"prime\":\"" << decimal(Field::PR) << "\",\"base_check_polynomial_degree\":" << cfg.weight * (1 + cfg.checks)
               << ",\"base_check_error_bound\":\"degree/(2^" << Field::PR_bit_len << "-1)\",\"protocol\":\"LeMans-chosen-single-point\","
               << "\"context_setup_seconds\":" << context_seconds
               << ",\"context_sent_bytes\":" << context_sent_bytes
               << ",\"context_received_bytes\":" << context_received_bytes
               << ",\"public_map_cached\":true,\"ot_mode\":\"" << mode << "\",\"network_debug_disclosure\":false,\"b\":" << cfg.output_length
               << ",\"t\":" << cfg.weight << ",\"log_block\":" << cfg.log_block
               << ",\"check_repetitions\":" << cfg.checks << ",\"sessions\":[";
        for (size_t session = 0; session < sessions; ++session) {
            Field delta;
            if (party == emp::ALICE) {
                do {
                    delta = uniform<Field>(entropy);
                    if (cheat == "force-cope-bit")
                        delta.val |= decltype(delta.val)(1) << (Field::PR_bit_len > 64 ? 64 : Field::PR_bit_len - 1);
                } while (delta == Field{} || delta.val >= Field::PR);
            }
            const auto start = std::chrono::steady_clock::now();
            const auto cpu_start = std::clock();
            const auto before = io.send_counter;
            const auto rounds_before = io.rounds;
            Output<Field> result;
            try {
                result = seeded_vole<Field, Packed>(io, party, delta, private_seed, cfg, session, connection_domain, entropy_source, cheat, persistent.get(), &public_map);
            } catch (...) {
                bool retired = false;
                try { (void)entropy_source.fork("after-protocol-abort"); }
                catch (const std::runtime_error&) { retired = true; }
                std::cerr << "entropy_retired=" << (retired ? "true" : "false") << '\n';
                if (!retired) throw std::runtime_error("entropy context survived protocol abort");
                throw;
            }
            const auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
            const double cpu_seconds = double(std::clock() - cpu_start) / CLOCKS_PER_SEC;
            if (session) output << ',';
            output << "{\"index\":" << session << ",\"seconds\":" << elapsed
                   << ",\"process_cpu_seconds\":" << cpu_seconds
                   << ",\"sent_bytes\":" << io.send_counter - before
                   << ",\"entropy_substreams_consumed\":" << entropy_source.substreams()
                   << ",\"direction_runs\":" << io.rounds - rounds_before;
            constexpr const char* phase_names[] = {"config", "base_vole", "tree_ot",
                "ggm_corrections", "checks", "public_map_generation", "public_map_multiply"};
            output << ",\"phase_seconds\":{";
            for (size_t phase = 0; phase < result.phase_seconds.size(); ++phase) {
                if (phase) output << ',';
                output << '\"' << phase_names[phase] << "\":" << result.phase_seconds[phase];
            }
            output << '}';
            constexpr const char* base_names[] = {"rot", "prg_allocation_reseed", "cope_extend", "check"};
            for (int measure = 0; measure < 2; ++measure) {
                output << (measure ? ",\"base_cpu_seconds\":{" : ",\"base_wall_seconds\":{");
                const auto& values = measure ? result.base_cpu_seconds : result.base_wall_seconds;
                for (size_t phase = 0; phase < values.size(); ++phase) {
                    if (phase) output << ',';
                    output << '\"' << base_names[phase] << "\":" << values[phase];
                }
                output << '}';
            }
            if (party == emp::ALICE) output << ",\"delta\":" << decimal(delta.val);
            else {
                output << ",\"value\":";
                write_fields(output, result.value);
            }
            output << ",\"share\":";
            write_fields(output, result.share);
            output << '}';
            std::cout << "role=" << party << " session=" << session
                      << " checks=passed security_reviewed=false seconds=" << elapsed
                      << " sent_bytes=" << io.send_counter - before << '\n';
        }
        output << "]}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "protocol aborted: " << error.what() << '\n';
        return 1;
    }
}

template<typename Field>
static int wire_test(int party, int port, const std::string& encoding) {
    emp::NetIO io(party == emp::ALICE ? nullptr : "127.0.0.1", port);
    const bool canonical = encoding == "canonical";
    if (!canonical && encoding != "modulus" && encoding != "highbit") return 2;
    if (party == emp::BOB) {
        auto raw = canonical ? Field::PR - 1 :
            (encoding == "modulus" ? Field::PR : Field::PR + 1);
        std::vector<unsigned char> bytes(Field::DS_byte_len);
        for (size_t byte = 0; byte < bytes.size(); ++byte) bytes[byte] = raw >> (8 * byte);
        io.send_data(bytes.data(), bytes.size());
        io.flush();
        return 0;
    }
    Field value;
    try { receive_fields(io, &value, 1); }
    catch (const std::runtime_error& error) {
        if (canonical) return 2;
        std::cerr << "noncanonical wire rejected: " << error.what() << '\n';
        return 1;
    }
    if (!canonical || value.val != Field::PR - 1) return 2;
    std::cout << "canonical field accepted\n";
    return 0;
}

int main(int argc, char** argv) {
    try { require_secure_randomness(); }
    catch (const std::exception& error) {
        std::cerr << "startup rejected: " << error.what() << '\n';
        return 2;
    }
    if (argc == 6 && std::string(argv[1]) == "wire-test") {
        const int party = std::stoi(argv[2]), port = std::stoi(argv[3]);
        if (party != 1 && party != 2) return 2;
        const std::string field = argv[4], encoding = argv[5];
        if (field == "fp61") return wire_test<FP61>(party, port, encoding);
        if (field == "fp127") return wire_test<FP127>(party, port, encoding);
        return 2;
    }
    if (argc == 5 && std::string(argv[1]) == "reflection") {
        const int party = std::stoi(argv[2]), port = std::stoi(argv[3]);
        const size_t size = std::stoul(argv[4]);
        if ((party != 1 && party != 2) || (size != 16 && size != 32)) return 2;
        emp::NetIO io(party == emp::ALICE ? nullptr : "127.0.0.1", port);
        if (party == emp::BOB) {
            Digest commitment;
            io.recv_data(commitment.data(), commitment.size());
            io.send_data(commitment.data(), commitment.size());
            io.flush();
            std::vector<unsigned char> opening(32 + size);
            io.recv_data(opening.data(), opening.size());
            io.send_data(opening.data(), opening.size());
            io.flush();
            return 0;
        }
        EndpointEntropy entropy_source;
        std::vector<unsigned char> value(size, 7);
        try {
            commit_exchange(io, party, entropy_source, value.data(), value.size(), "reflection-test/connection-1");
        } catch (const std::exception& error) {
            bool retired = false;
            try { (void)entropy_source.fork("after-abort"); }
            catch (const std::runtime_error&) { retired = true; }
            if (!retired) {
                std::cerr << "entropy context unexpectedly survived abort\n";
                return 2;
            }
            std::cerr << "reflection rejected: " << error.what() << '\n';
            return 1;
        }
        std::cerr << "reflection unexpectedly accepted\n";
        return 0;
    }
    if (argc < 5) {
        std::cerr << "usage: seeded_vole <1|2> <port> <output.json> <sessions> [cheat] [b] [t] [log_block] [persistent|fresh] [fp61|fp127]\n";
        return 2;
    }
    const std::string field = argc > 10 ? argv[10] : "fp61";
    if (field == "fp61") return run_field<FP61, FP61x2>(argc, argv, "fp61");
    if (field == "fp127") return run_field<FP127, FP127x2>(argc, argv, "fp127");
    std::cerr << "unknown field\n";
    return 2;
}
