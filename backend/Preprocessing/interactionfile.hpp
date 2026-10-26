#ifndef __EXASIM_INTERACTION_FILE_HPP__
#define __EXASIM_INTERACTION_FILE_HPP__

#include <algorithm>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

struct ParsedInteractionInput {
    std::vector<double> table;
    std::vector<int> selectors;
    int nfint = 1;
    int maxcondition = 0;
};

inline std::string interactionTrim(const std::string& value) {
    const auto first = value.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return "";
    return value.substr(first, value.find_last_not_of(" \t\r\n") - first + 1);
}

inline std::vector<int> interactionIntegers(std::string value, bool vectorValue) {
    value = interactionTrim(value);
    if (vectorValue) {
        if (value.size() < 2 || value.front() != '[' || value.back() != ']')
            throw std::runtime_error("interaction.txt vector values must be enclosed in brackets");
        value = value.substr(1, value.size()-2);
    } else if (value.find_first_of("[]") != std::string::npos) {
        throw std::runtime_error("interaction.txt scalar values must be single integers");
    }
    for (char& ch : value) if (ch == ',' || ch == ';') ch = ' ';
    std::istringstream input(value); std::vector<int> result; std::string token;
    while (input >> token) {
        if (!token.empty() && token.front() == '+')
            throw std::runtime_error("interaction.txt contains an invalid integer");
        std::size_t used=0; long long number=0;
        try { number=std::stoll(token,&used); } catch (...) { throw std::runtime_error("interaction.txt contains a non-integer token"); }
        if (used!=token.size() || number<0 || number>=(1LL<<31))
            throw std::runtime_error("interaction.txt contains an invalid integer");
        result.push_back(static_cast<int>(number));
    }
    if (result.empty()) throw std::runtime_error("interaction.txt contains an empty value");
    return result;
}

inline ParsedInteractionInput parseInteractionFileStrict(const std::string& filename,
    int numdomains, int localdomain, int ncu,
    const std::vector<int>& boundaryconditions, int boundaryexprcount)
{
    std::ifstream file(filename); if (!file) throw std::runtime_error("Unable to open interaction file " + filename);
    static const std::vector<std::string> keys={"domains","type","interface1","condition1","fluxmap1","interface2","condition2","fluxmap2"};
    std::vector<std::pair<std::string,std::string>> statements; std::string line,buffer;
    while (std::getline(file,line)) {
        const auto first=line.find_first_not_of(" \t\r");
        if (first==std::string::npos || line.compare(first,2,"//")==0) continue;
        buffer += line;
        std::size_t semi;
        while ((semi=buffer.find(';'))!=std::string::npos) {
            const std::string statement=buffer.substr(0,semi); buffer.erase(0,semi+1);
            const auto equals=statement.find('=');
            if (equals==std::string::npos || statement.find('=',equals+1)!=std::string::npos)
                throw std::runtime_error("interaction.txt statement must contain one '='");
            statements.emplace_back(interactionTrim(statement.substr(0,equals)),interactionTrim(statement.substr(equals+1)));
        }
    }
    if (!interactionTrim(buffer).empty() || statements.empty() || statements.size()%8!=0)
        throw std::runtime_error("interaction.txt has an incomplete interaction record");
    if (numdomains<2 || localdomain<1 || localdomain>numdomains) throw std::runtime_error("invalid interaction domain metadata");
    ParsedInteractionInput result; const int count=static_cast<int>(statements.size()/8);
    result.table={1.0,static_cast<double>(numdomains),static_cast<double>(count)}; int width=0;
    for (int k=0;k<count;++k) {
        std::vector<std::vector<int>> v(8);
        for (int j=0;j<8;++j) {
            const auto& item=statements[static_cast<std::size_t>(8*k+j)];
            if (item.first!=keys[j]) throw std::runtime_error("interaction.txt keys are unknown, repeated, or out of order");
            v[j]=interactionIntegers(item.second,j==0||j==4||j==7);
            if (j!=0&&j!=4&&j!=7&&v[j].size()!=1) throw std::runtime_error("interaction.txt scalar field must contain one integer");
        }
        if (v[0].size()!=2 || v[0][0]<1 || v[0][1]<1 || v[0][0]>numdomains || v[0][1]>numdomains || v[0][0]==v[0][1])
            throw std::runtime_error("interaction.txt domains field is invalid");
        if (v[1][0]!=1) throw std::runtime_error("generalized interactions support only type=1");
        for (int side=0;side<2;++side) {
            const int si=side?5:2, ci=side?6:3, mi=side?7:4;
            if (v[si][0]<1 || v[ci][0]<1 || v[mi].empty()) throw std::runtime_error("invalid interaction endpoint data");
            if (v[0][side]==localdomain) {
                const int selector=v[si][0];
                if (selector>boundaryexprcount || selector>static_cast<int>(boundaryconditions.size())) throw std::runtime_error("interaction selector is outside the local boundary list");
                const int boundary=boundaryconditions[selector-1];
                if (boundary<=0 || std::count(boundaryconditions.begin(),boundaryconditions.end(),boundary)!=1) throw std::runtime_error("interaction selector has a nonunique boundary condition");
                if (std::find(result.selectors.begin(),result.selectors.end(),selector)!=result.selectors.end()) throw std::runtime_error("interaction selector overlaps another interaction");
                for (int component:v[mi]) if (component<1 || component>ncu) throw std::runtime_error("interaction flux map is outside ncu");
                result.selectors.push_back(selector); result.maxcondition=std::max(result.maxcondition,v[ci][0]);
                const int donated=static_cast<int>(v[side?4:7].size());
                if (width!=0 && width!=donated) throw std::runtime_error("interactions require one fint output width per domain");
                width=donated; result.nfint=donated;
            }
        }
        const int n1=v[4].size(),n2=v[7].size(); result.table.insert(result.table.end(),{static_cast<double>(9+n1+n2),static_cast<double>(v[0][0]-1),static_cast<double>(v[0][1]-1),1.0});
        for (int side=0;side<2;++side) { const int si=side?5:2,ci=side?6:3,mi=side?7:4; result.table.insert(result.table.end(),{static_cast<double>(v[si][0]),static_cast<double>(v[ci][0]),static_cast<double>(v[mi].size())}); for(int x:v[mi]) result.table.push_back(x-1); }
    }
    return result;
}
#endif
