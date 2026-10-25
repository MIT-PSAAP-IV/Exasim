#pragma once

#include <algorithm>
#include <vector>

namespace exasim::preprocessing {

inline void interface_elements_for_selectors(std::vector<int>& elements,
                                             std::vector<int>& localFaces,
                                             const std::vector<int>& faceSelectors,
                                             int facesPerElement, int elementCount,
                                             const std::vector<int>& selectors)
{
    elements.clear();
    localFaces.clear();
    for (int elem = 0; elem < elementCount; ++elem)
        for (int face = 0; face < facesPerElement; ++face)
            if (std::find(selectors.begin(), selectors.end(),
                          faceSelectors[face + facesPerElement*elem]) != selectors.end()) {
                elements.push_back(elem);
                localFaces.push_back(face);
            }
}

inline std::vector<int> unique_interface_elements(const std::vector<int>& faceElements)
{
    std::vector<int> result(faceElements);
    std::sort(result.begin(), result.end());
    result.erase(std::unique(result.begin(), result.end()), result.end());
    return result;
}

} // namespace exasim::preprocessing
