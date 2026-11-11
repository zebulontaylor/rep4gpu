// SPDX-License-Identifier: Apache-2.0
// Built on fp127.h, adapted from vole-labs/vole FP107 at
// e266595c1811692a6b2df87a4b4d13d3a86ef80f. Upstream:
// https://github.com/vole-labs/vole; see VOLE-LICENSE.
#ifndef SIV_MATRIX_VOPE_FP127EXT4_H
#define SIV_MATRIX_VOPE_FP127EXT4_H

#include "fp127.h"
#include <array>
#include <cstddef>
#include <stdexcept>

// F_(p^4) in the tower i^2=-1, v^2=2+i, p=2^127-1.
// Coordinate order is [1,i,v,i*v]. Since p=3 mod4, i^2+1 is irreducible.
// Norm(2+i)=5 is a nonsquare in F_p (p=2 mod5), so v^2-(2+i) is
// irreducible over F_(p^2). The equivalent polynomial is v^4-4v^2+5.
class FP127Ext4 {
 public:
    using Coordinates = std::array<FP127,4>;
    using u128 = FP127::u128;
    static constexpr std::size_t ENCODED_SIZE = 64;

    FP127Ext4() : coefficients_{} {}
    explicit FP127Ext4(const FP127& value)
        : coefficients_{{value,FP127(),FP127(),FP127()}} { validate(); }
    explicit FP127Ext4(const Coordinates& values) : coefficients_(values) { validate(); }
    FP127Ext4(const FP127& a0,const FP127& a1,const FP127& a2,const FP127& a3)
        : coefficients_{{a0,a1,a2,a3}} { validate(); }

    static FP127Ext4 embed(const FP127& value) { return FP127Ext4(value); }
    static FP127Ext4 basis_unit(std::size_t index) {
        if (index>=4) throw std::out_of_range("F127^4 basis index must be below 4");
        Coordinates out{};
        out[index]=FP127(1);
        return FP127Ext4(out);
    }
    const Coordinates& coordinates() const { return coefficients_; }
    const Coordinates& coefficients() const { return coefficients_; }
    const FP127& operator[](std::size_t index) const { return coefficients_.at(index); }

    bool operator==(const FP127Ext4& other) const { return coefficients_==other.coefficients_; }
    bool operator!=(const FP127Ext4& other) const { return !(*this==other); }
    bool is_zero() const {
        for (const auto& value:coefficients_) if (value.value()!=0) return false;
        return true;
    }
    FP127Ext4 operator+(const FP127Ext4& other) const {
        return FP127Ext4(coefficients_[0]+other[0],coefficients_[1]+other[1],
                         coefficients_[2]+other[2],coefficients_[3]+other[3]);
    }
    FP127Ext4 operator-(const FP127Ext4& other) const {
        return FP127Ext4(coefficients_[0]-other[0],coefficients_[1]-other[1],
                         coefficients_[2]-other[2],coefficients_[3]-other[3]);
    }
    FP127Ext4 negate() const {
        return FP127Ext4(coefficients_[0].negate(),coefficients_[1].negate(),
                         coefficients_[2].negate(),coefficients_[3].negate());
    }
    FP127Ext4 operator-() const { return negate(); }
    FP127Ext4 operator*(const FP127Ext4& other) const {
        const Quad a{coefficients_[0],coefficients_[1]}, b{coefficients_[2],coefficients_[3]};
        const Quad c{other[0],other[1]}, d{other[2],other[3]};
        const Quad ac=mul(a,c),bd=mul(b,d);
        const Quad low=add(ac,times_alpha(bd));
        const Quad high=sub(sub(mul(add(a,b),add(c,d)),ac),bd);
        return FP127Ext4(low.r,low.i,high.r,high.i);
    }
    FP127Ext4 operator*(const FP127& scalar) const {
        if (scalar.value()>=FP127::PR) throw std::invalid_argument("noncanonical F127 scalar");
        return FP127Ext4(coefficients_[0]*scalar,coefficients_[1]*scalar,
                         coefficients_[2]*scalar,coefficients_[3]*scalar);
    }
    friend FP127Ext4 operator*(const FP127& scalar,const FP127Ext4& value) { return value*scalar; }
    FP127Ext4& operator+=(const FP127Ext4& other) { *this=*this+other;return *this; }
    FP127Ext4& operator-=(const FP127Ext4& other) { *this=*this-other;return *this; }
    FP127Ext4& operator*=(const FP127Ext4& other) { *this=*this*other;return *this; }
    FP127Ext4& operator*=(const FP127& scalar) { *this=*this*scalar;return *this; }

    FP127Ext4 inv() const {
        if (is_zero()) throw std::domain_error("zero has no F127^4 inverse");
        const Quad a{coefficients_[0],coefficients_[1]}, b{coefficients_[2],coefficients_[3]};
        const Quad denominator=sub(mul(a,a),times_alpha(mul(b,b)));
        const FP127 norm=denominator.r*denominator.r+denominator.i*denominator.i;
        if (norm.value()==0) throw std::logic_error("invalid F127^4 norm");
        const FP127 reciprocal=norm.inv();
        const Quad inverse_denominator{denominator.r*reciprocal,denominator.i.negate()*reciprocal};
        const Quad low=mul(a,inverse_denominator),high=mul(b,inverse_denominator);
        return FP127Ext4(low.r,low.i,high.r.negate(),high.i.negate());
    }
    FP127Ext4 operator/(const FP127Ext4& other) const { return *this*other.inv(); }
    FP127Ext4& operator/=(const FP127Ext4& other) { *this=*this/other;return *this; }
    FP127Ext4 pow(u128 exponent) const {
        FP127Ext4 result(FP127(1)),base=*this;
        while (exponent) {
            if (exponent&1) result*=base;
            exponent>>=1;
            if (exponent) base*=base;
        }
        return result;
    }

    template<typename PRNG>
    void rand(PRNG& prg) {
        // Each coordinate independently uses FP127's 127-bit rejection sampler.
        Coordinates fresh{};
        for (auto& value:fresh) value.rand(prg);
        coefficients_=fresh;
    }
    template<typename PRNG>
    static FP127Ext4 uniform(PRNG& prg) { FP127Ext4 value;value.rand(prg);return value; }

    std::array<unsigned char,ENCODED_SIZE> encode() const {
        validate();
        std::array<unsigned char,ENCODED_SIZE> out{};
        for (std::size_t h=0;h<4;++h) {
            const auto bytes=coefficients_[h].encode();
            for (std::size_t j=0;j<16;++j) out[16*h+j]=bytes[j];
        }
        return out;
    }
    static FP127Ext4 decode(const unsigned char* bytes,std::size_t length) {
        if (length!=ENCODED_SIZE || bytes==nullptr)
            throw std::invalid_argument("F127^4 encoding must contain exactly 64 bytes");
        Coordinates values{};
        for (std::size_t h=0;h<4;++h) values[h]=FP127::decode(bytes+16*h);
        return FP127Ext4(values);
    }
    static FP127Ext4 decode(const std::array<unsigned char,ENCODED_SIZE>& bytes) {
        return decode(bytes.data(),bytes.size());
    }
    template<typename IO>
    void send(IO* io) const {
        const auto bytes=encode();
        io->send_data(bytes.data(),bytes.size());
    }
    template<typename IO>
    void recv(IO* io) {
        std::array<unsigned char,ENCODED_SIZE> bytes;
        io->recv_data(bytes.data(),bytes.size());
        const auto decoded=decode(bytes);
        *this=decoded;
    }
    static constexpr std::size_t size() { return ENCODED_SIZE; }

 private:
    struct Quad { FP127 r,i; };
    Coordinates coefficients_;
    void validate() const {
        for (const auto& value:coefficients_)
            if (value.value()>=FP127::PR) throw std::invalid_argument("noncanonical F127^4 coordinate");
    }
    static Quad add(const Quad& a,const Quad& b) { return {a.r+b.r,a.i+b.i}; }
    static Quad sub(const Quad& a,const Quad& b) { return {a.r-b.r,a.i-b.i}; }
    static Quad mul(const Quad& a,const Quad& b) {
        const FP127 real_product=a.r*b.r,imaginary_product=a.i*b.i;
        return {real_product-imaginary_product,(a.r+a.i)*(b.r+b.i)-real_product-imaginary_product};
    }
    static Quad times_alpha(const Quad& a) { return {a.r+a.r-a.i,a.r+a.i+a.i}; }
};

using FP1274 = FP127Ext4;

#endif  // SIV_MATRIX_VOPE_FP127EXT4_H
