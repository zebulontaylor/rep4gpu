#include "spvope_candidate.hpp"
#include "fp127x2.h"
#include <fstream>
#include <iostream>

using namespace matrix_vope;

template<typename Integer> static std::string decimal(Integer value) {
    std::string out;
    do { out.push_back(char('0' + value % 10)); value /= 10; } while (value);
    std::reverse(out.begin(), out.end());
    return out;
}
template<typename Field> static void write_fields(std::ostream& out, const std::vector<Field>& values) {
    out << '[';
    for (size_t i = 0; i < values.size(); ++i) {
        if (i) out << ',';
        out << decimal(values[i].val);
    }
    out << ']';
}

template<typename Field, typename Packed>
static int run(int argc, char** argv, const std::string& field) {
    try {
        const int party = std::stoi(argv[1]), port = std::stoi(argv[2]);
        const size_t sessions = std::stoul(argv[4]);
        if ((party != 1 && party != 2) || sessions < 1 || sessions > 32)
            throw std::invalid_argument("invalid role or session count");
        const std::string fault = argc > 5 ? argv[5] : "none";
        const std::string beta_mode = argc > 6 ? argv[6] : "uniform";
        const std::string backend = argc > 7 ? argv[7] : "pcg";
        if (backend != "pcg" && backend != "direct") throw std::invalid_argument("unknown backend");
        if (fault != "none" && fault != "tree" && fault != "offset" && fault != "mac" &&
            fault != "check" && fault != "seed" && fault != "selective")
            throw std::invalid_argument("unknown fault");
        const size_t a = argc > 9 ? std::stoul(argv[9]) : 16;
        Config cfg;
        if (argc > 10) cfg.output_length = std::stoul(argv[10]);
        emp::NetIO io(party == 1 ? nullptr : "127.0.0.1", port);
        EndpointEntropy entropy;
        auto connection = joint_coin(io, party, entropy, "spvope-candidate/connection");
        auto tag = digest(&connection, sizeof(connection));
        std::string domain = "spvope-candidate/";
        const char* hex = "0123456789abcdef";
        for (auto byte : tag) { domain.push_back(hex[byte >> 4]); domain.push_back(hex[byte & 15]); }
        SpvopeContext<Field> context(io, party);
        auto inputs = entropy.fork(domain + "/private-inputs");
        emp::block right_seed;
        inputs.random_block(&right_seed, 1);
        std::ofstream output(argv[3]);
        if (!output) throw std::runtime_error("cannot create role output");
        output << "{\"role\":" << party << ",\"field\":\"" << field
               << "\",\"prime_bits\":" << Field::PR_bit_len << ",\"a\":" << a
               << ",\"b\":" << cfg.output_length << ",\"backend\":\"" << backend
               << "\",\"security_reviewed\":false,\"malicious_privacy\":false,\"arbitrary_beta_malicious_privacy\":false,"
               << "\"network_debug_disclosure\":false,\"relation\":\"U-W=e*v^T\",\"sessions\":[";
        for (size_t session = 0; session < sessions; ++session) {
            size_t alpha = 0;
            Field beta{};
            if (party == 1) {
                uint64_t position;
                inputs.random_data_unaligned(&position, sizeof(position));
                alpha = position & (a - 1);
                if (beta_mode == "uniform") {
                    do { beta = uniform<Field>(inputs); } while (beta == Field{});
                } else if (beta_mode == "zero") beta = Field(0);
                else if (beta_mode == "one") beta = Field(1);
                else if (beta_mode == "two") beta = Field(2);
                else throw std::invalid_argument("unknown private beta mode");
            }
            const auto start = std::chrono::steady_clock::now();
            const auto bytes = io.send_counter;
            SpvopeOutput<Field> result;
            try {
                result = candidate_spvope<Field, Packed>(io, party, a, alpha, beta,
                    right_seed, cfg, session, domain, entropy, context, backend == "direct", fault);
            } catch (...) {
                bool retired = false;
                try { (void)entropy.fork("after-abort"); }
                catch (const std::runtime_error&) { retired = true; }
                std::cerr << "entropy_retired=" << (retired ? "true" : "false") << '\n';
                throw;
            }
            double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
            if (session) output << ',';
            output << "{\"index\":" << session << ",\"seconds\":" << elapsed
                   << ",\"sent_bytes\":" << io.send_counter - bytes;
            if (party == 1) output << ",\"alpha\":" << alpha << ",\"beta\":" << decimal(beta.val)
                << ",\"first_delta\":" << decimal(result.first_delta.val)
                << ",\"second_delta\":" << decimal(result.second_delta.val);
            else { output << ",\"v\":"; write_fields(output, result.right_vector); }
            output << ",\"matrix_share\":"; write_fields(output, result.matrix_share); output << '}';
            std::cout << "role=" << party << " session=" << session
                      << " checked_relation=true security_reviewed=false\n";
        }
        output << "]}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "candidate aborted: " << error.what() << '\n';
        return 1;
    }
}

int main(int argc, char** argv) {
    try { require_secure_randomness(); }
    catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 2; }
    if (argc < 5) {
        std::cerr << "usage: spvope_candidate role port output sessions [fault] [beta_mode] [pcg|direct] [fp61|fp127] [a] [b]\n";
        return 2;
    }
    std::string field = argc > 8 ? argv[8] : "fp127";
    if (field == "fp61") return run<FP61, FP61x2>(argc, argv, field);
    if (field == "fp127") return run<FP127, FP127x2>(argc, argv, field);
    return 2;
}
