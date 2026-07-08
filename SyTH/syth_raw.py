import ctypes

# _lib_path = os.path.join(os.path.dirname(__file__), "lib", "libsyth.so")
_lib = ctypes.CDLL("/root/syth/build/syTH/syTH_driver/libsyTH.so")

SYTH_API_VERSION = 1  # match C++ macro
SRK = ctypes.c_double
SIK = ctypes.c_int


def _bind_void_function(name, argtypes):
    function = getattr(_lib, name)
    function.argtypes = argtypes
    function.restype = None
    return function


# Mirror each extern "C" function
syTH_initialize = _bind_void_function(
    "syTH_initialize", [ctypes.c_char_p, ctypes.c_size_t, ctypes.c_int]
)
syTH_writeSolution = _bind_void_function(
    "syTH_writeSolution", [ctypes.c_char_p, ctypes.c_size_t]
)
syTH_writeSolutionToInputFile = _bind_void_function("syTH_writeSolutionToInputFile", [])
syTH_finalize = _bind_void_function("syTH_finalize", [])
syTH_solveFluidFlow = _bind_void_function(
    "syTH_solveFluidFlow", [ctypes.c_double, ctypes.c_double]
)
syTH_solveMassTransport = _bind_void_function(
    "syTH_solveMassTransport", [ctypes.c_double, ctypes.c_double]
)
syTH_specifyCurrentCorePower = _bind_void_function(
    "syTH_specifyCurrentCorePower", [ctypes.c_double]
)
syTH_setDepletionTime = _bind_void_function("syTH_setDepletionTime", [ctypes.c_double])
syTH_setFluidSurfaceValue = _bind_void_function(
    "syTH_setFluidSurfaceValue",
    [ctypes.c_double, ctypes.c_char_p, ctypes.c_size_t, ctypes.c_int],
)
syTH_setInletValue = _bind_void_function(
    "syTH_setInletValue", [ctypes.c_double, ctypes.c_char_p, ctypes.c_size_t]
)
syTH_setOutletValue = _bind_void_function(
    "syTH_setOutletValue", [ctypes.c_double, ctypes.c_char_p, ctypes.c_size_t]
)
syTH_setNodePower = _bind_void_function(
    "syTH_setNodePower", [ctypes.c_int, ctypes.c_double]
)
syTH_setNodeFissionRxRate = _bind_void_function(
    "syTH_setNodeFissionRxRate", [ctypes.c_int, ctypes.c_int, ctypes.c_double]
)
syTH_addNodePower = _bind_void_function(
    "syTH_addNodePower", [ctypes.c_int, ctypes.c_double]
)
syTH_setNeutronPrecursorSourceCoupledToPowerFactor = _bind_void_function(
    "syTH_setNeutronPrecursorSourceCoupledToPowerFactor", [ctypes.c_double]
)

syTH_setSpeciesSource = _bind_void_function(
    "syTH_setSpeciesSource", [ctypes.c_int, ctypes.c_int, ctypes.c_double]
)
syTH_setSpeciesDecayConstant = _bind_void_function(
    "syTH_setSpeciesDecayConstant", [ctypes.c_int, ctypes.c_double]
)
syTH_setBurnup = _bind_void_function("syTH_setBurnup", [ctypes.c_int, ctypes.c_double])
syTH_setState = _bind_void_function("syTH_setState", [ctypes.c_int])
syTH_getNodeCount = _bind_void_function("syTH_getNodeCount", [ctypes.POINTER(SIK)])
syTH_getNodeBounds = _bind_void_function(
    "syTH_getNodeBounds", [ctypes.c_int, ctypes.POINTER(SRK)]
)
syTH_getNodeVolume = _bind_void_function(
    "syTH_getNodeVolume", [ctypes.c_int, ctypes.POINTER(SRK)]
)
syTH_getFluidSurfaceValue = _bind_void_function(
    "syTH_getFluidSurfaceValue",
    [ctypes.c_char_p, ctypes.c_size_t, ctypes.c_int, ctypes.POINTER(SRK)],
)
syTH_getFluidInletValue = _bind_void_function(
    "syTH_getFluidInletValue", [ctypes.c_char_p, ctypes.c_size_t, ctypes.POINTER(SRK)]
)
syTH_getFluidOutletValue = _bind_void_function(
    "syTH_getFluidOutletValue", [ctypes.c_char_p, ctypes.c_size_t, ctypes.POINTER(SRK)]
)
syTH_getFluidNodeValue = _bind_void_function(
    "syTH_getFluidNodeValue",
    [ctypes.c_char_p, ctypes.c_size_t, ctypes.c_int, ctypes.POINTER(SRK)],
)
syTH_addSpecies = _bind_void_function(
    "syTH_addSpecies", [ctypes.c_char_p, ctypes.c_size_t, ctypes.POINTER(SIK)]
)
syTH_Get_Mod_Node_Avg_Temp = _bind_void_function(
    "syTH_Get_Mod_Node_Avg_Temp", [ctypes.c_int, ctypes.POINTER(SRK)]
)
syTH_getIterations = _bind_void_function("syTH_getIterations", [ctypes.POINTER(SIK)])
syTH_checkInit = _bind_void_function("syTH_checkInit", [ctypes.POINTER(ctypes.c_bool)])
syTH_getNumStates = _bind_void_function("syTH_getNumStates", [ctypes.POINTER(SIK)])
syTH_checkCurrentCorePower = _bind_void_function(
    "syTH_checkCurrentCorePower", [ctypes.POINTER(SRK)]
)
syTH_getDepletionTime = _bind_void_function(
    "syTH_getDepletionTime", [ctypes.POINTER(SRK)]
)
syTH_getBurnup = _bind_void_function(
    "syTH_getBurnup", [ctypes.c_int, ctypes.POINTER(SRK)]
)
syTH_getPressureDifferential = _bind_void_function(
    "syTH_getPressureDifferential", [ctypes.POINTER(SRK)]
)
syTH_getSpecies = _bind_void_function(
    "syTH_getSpecies", [ctypes.c_int, ctypes.c_int, ctypes.POINTER(SRK)]
)
syTH_getTotalDelayedNeutronSources = _bind_void_function(
    "syTH_getTotalDelayedNeutronSources", [ctypes.c_int, ctypes.POINTER(SRK)]
)
