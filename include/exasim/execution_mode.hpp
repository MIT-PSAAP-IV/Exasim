#pragma once

enum class ExasimExecutionMode {
    Solve,
    Postprocess,
    AuxiliaryHelmholtz,
    AuxiliaryElasticity,
    HDGLDGSetup // Allocate native workspaces; only the coupled driver may execute.
};
