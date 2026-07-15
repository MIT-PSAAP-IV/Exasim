#pragma once

// Angle-bracket (not a source-relative path): resolves against the include search path,
// so it works in BOTH the source-tree layout (backend/ sibling of include/, -I <root>/include)
// AND an install layout (backend nested under include/, -I <prefix>/include). The old
// "../../../include/driver_abi.hpp" assumed the source layout and broke install-based builds
// of the operator-export / preprocessing path (e.g. a text2code --emit-app app). driver_abi.hpp
// sits at the same include root as <exasim/...>, so it resolves wherever <exasim/operators.hpp>
// does — the only context that reaches this header.
#include <driver_abi.hpp>
