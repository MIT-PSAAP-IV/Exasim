#ifndef __NONMATCHING_INTERACTION_HPP__
#define __NONMATCHING_INTERACTION_HPP__

#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace exasim {
namespace nonmatching {

template <class I>
struct InteractionEndpointRecord {
    I domain = 0;
    I interfaceSelector = 0;
    I condition = 0;
    std::vector<I> fluxmap;
};

template <class I>
struct InteractionRecord {
    I ordinal = 0;
    I type = 0;
    InteractionEndpointRecord<I> endpoint[2];
};

// Non-owning graph-only inputs to one invocation of the established pairwise
// setup.  The owning NonmatchingInteraction copies fluxmap before this
// descriptor goes out of scope.
template <class I>
struct NonmatchingEndpoint {
    I side = 0;
    I boundary = 0;
    I receivedWidth = 0;
    I donatedWidth = 0;
    const I* fluxmap = nullptr;
};

template <class I>
inline I checked_interaction_tag(I ordinal, I direction, I upperBound)
{
    if (ordinal < 0 || direction < 0 || direction > 1 || upperBound < 0)
        throw std::out_of_range("invalid interaction message tag input");
    const unsigned long long value = 27100ULL +
            2ULL * static_cast<unsigned long long>(ordinal) +
            static_cast<unsigned long long>(direction);
    if (value > static_cast<unsigned long long>(upperBound) ||
            value > static_cast<unsigned long long>(std::numeric_limits<I>::max()))
        throw std::out_of_range("interaction message tag exceeds its upper bound");
    return static_cast<I>(value);
}

template <class I>
inline I checked_interaction_integer(double value, const char *what)
{
    if (!std::isfinite(value) || value < 0.0 || std::floor(value) != value ||
        value > static_cast<double>(std::numeric_limits<I>::max()))
        throw std::invalid_argument(std::string("invalid interaction table ") + what);
    return static_cast<I>(value);
}

template <class I>
inline std::vector<InteractionRecord<I>>
parse_interaction_table(const double *raw, std::size_t length)
{
    std::vector<InteractionRecord<I>> records;
    if (length == 0) return records;
    if (raw == nullptr || length < 3)
        throw std::invalid_argument("interaction table is shorter than its header");

    std::vector<I> table;
    table.reserve(length);
    for (std::size_t i = 0; i < length; ++i)
        table.push_back(checked_interaction_integer<I>(raw[i], "value"));

    if (table[0] != static_cast<I>(1))
        throw std::invalid_argument("unsupported interaction table version");
    const I domainCount = table[1];
    const I recordCount = table[2];
    if (domainCount < static_cast<I>(2) || recordCount < static_cast<I>(1))
        throw std::invalid_argument("invalid interaction table header");
    if (static_cast<unsigned long long>(recordCount) >
        static_cast<unsigned long long>(length))
        throw std::invalid_argument("interaction record count exceeds table length");

    std::size_t cursor = 3;
    records.reserve(static_cast<std::size_t>(recordCount));
    for (I ordinal = 0; ordinal < recordCount; ++ordinal) {
        if (cursor >= length)
            throw std::invalid_argument("interaction table is missing a record length");
        const std::size_t recordLength = static_cast<std::size_t>(table[cursor++]);
        if (recordLength < 11 || recordLength > length - cursor)
            throw std::invalid_argument("invalid interaction record length");
        const std::size_t recordEnd = cursor + recordLength;

        InteractionRecord<I> record;
        record.ordinal = ordinal;
        record.endpoint[0].domain = table[cursor++];
        record.endpoint[1].domain = table[cursor++];
        record.type = table[cursor++];
        if (record.endpoint[0].domain >= domainCount ||
            record.endpoint[1].domain >= domainCount ||
            record.endpoint[0].domain == record.endpoint[1].domain)
            throw std::invalid_argument("invalid interaction endpoint domains");
        if (record.type != static_cast<I>(1))
            throw std::invalid_argument("generalized interactions support only type=1");

        for (int side = 0; side < 2; ++side) {
            if (recordEnd - cursor < 3)
                throw std::invalid_argument("truncated interaction endpoint");
            auto &endpoint = record.endpoint[side];
            endpoint.interfaceSelector = table[cursor++];
            endpoint.condition = table[cursor++];
            const std::size_t mapLength = static_cast<std::size_t>(table[cursor++]);
            if (endpoint.interfaceSelector < static_cast<I>(1) ||
                endpoint.condition < static_cast<I>(1) || mapLength == 0 ||
                mapLength > recordEnd - cursor)
                throw std::invalid_argument("invalid interaction endpoint data");
            endpoint.fluxmap.assign(table.begin() + static_cast<std::ptrdiff_t>(cursor),
                                    table.begin() + static_cast<std::ptrdiff_t>(cursor + mapLength));
            cursor += mapLength;
        }
        if (cursor != recordEnd)
            throw std::invalid_argument("interaction record length does not match its maps");
        records.push_back(std::move(record));
    }
    if (cursor != length)
        throw std::invalid_argument("interaction table has trailing values");
    return records;
}

template <class I, typename std::enable_if<!std::is_same<I, double>::value, int>::type = 0>
inline std::vector<InteractionRecord<I>>
parse_interaction_table(const I *raw, std::size_t length)
{
    if (length == 0) return {};
    if (!raw) throw std::invalid_argument("interaction table is shorter than its header");
    std::vector<double> values(length);
    for (std::size_t i = 0; i < length; ++i)
        values[i] = static_cast<double>(raw[i]);
    return parse_interaction_table<I>(values.data(), values.size());
}

} // namespace nonmatching
} // namespace exasim

#endif
