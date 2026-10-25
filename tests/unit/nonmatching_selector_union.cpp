#include "backend/Preprocessing/selectorunion.hpp"

#include <cstdlib>
#include <iostream>
#include <vector>

namespace {
void require(bool condition, const char* message)
{
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}
}

int main()
{
    // Column-major [face + nfe*element]. Element 1 owns two selected faces.
    const std::vector<int> faceSelectors{
        0, 1, 3, 4,
        1, 3, 2, 4,
        0, 2, 3, 4};
    std::vector<int> elements, localFaces;

    exasim::preprocessing::interface_elements_for_selectors(
        elements, localFaces, faceSelectors, 4, 3, {});
    require(elements.empty() && localFaces.empty(), "empty selector union selected faces");

    exasim::preprocessing::interface_elements_for_selectors(
        elements, localFaces, faceSelectors, 4, 3, {1});
    require(elements == std::vector<int>({0, 1}), "one-selector face elements differ");
    require(localFaces == std::vector<int>({1, 0}), "one-selector face order differs");

    exasim::preprocessing::interface_elements_for_selectors(
        elements, localFaces, faceSelectors, 4, 3, {1, 3});
    require(elements == std::vector<int>({0, 0, 1, 1, 2}), "union face records differ");
    require(localFaces == std::vector<int>({1, 2, 0, 1, 2}), "union traversal order differs");
    require(exasim::preprocessing::unique_interface_elements(elements) ==
            std::vector<int>({0, 1, 2}), "interface element union was not deduplicated");

    std::cout << "PASS: selector-union face records and unique elements\n";
    return 0;
}
