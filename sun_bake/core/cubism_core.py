import os
import ctypes

CSM_ALIGN_OF_MOC = 64
CSM_ALIGN_OF_MODEL = 16


class csmVector2(ctypes.Structure):
    _fields_ = [
        ("X", ctypes.c_float),
        ("Y", ctypes.c_float),
    ]


class csmVector4(ctypes.Structure):
    _fields_ = [
        ("X", ctypes.c_float),
        ("Y", ctypes.c_float),
        ("Z", ctypes.c_float),
        ("W", ctypes.c_float),
    ]


def find_cubism_dll(custom_path=None):
    if custom_path and os.path.isfile(custom_path):
        return os.path.abspath(custom_path)

    env_path = os.environ.get("CUBISM_CORE_DLL")
    if env_path and os.path.isfile(env_path):
        return os.path.abspath(env_path)

    main_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    dll_path = os.path.join(main_dir, "Live2DCubismCore.dll")
    if os.path.isfile(dll_path):
        return os.path.abspath(dll_path)

    cwd_path = os.path.join(os.getcwd(), "Live2DCubismCore.dll")
    if os.path.isfile(cwd_path):
        return os.path.abspath(cwd_path)

    raise FileNotFoundError(
        "Live2DCubismCore.dll not found. Please place Live2DCubismCore.dll directly next to main.py."
    )


class CubismCore:
    _instance = None

    def __init__(self, dll_path=None):
        self.dll_path = find_cubism_dll(dll_path)
        self._dll = ctypes.CDLL(self.dll_path)
        self._setup_bindings()

    @classmethod
    def get_instance(cls, dll_path=None):
        if cls._instance is None:
            cls._instance = cls(dll_path)
        return cls._instance

    def _setup_bindings(self):
        d = self._dll

        d.csmGetVersion.restype = ctypes.c_uint
        d.csmGetLatestMocVersion.restype = ctypes.c_uint
        d.csmGetMocVersion.restype = ctypes.c_uint
        d.csmGetMocVersion.argtypes = [ctypes.c_void_p, ctypes.c_uint]
        d.csmHasMocConsistency.restype = ctypes.c_int
        d.csmHasMocConsistency.argtypes = [ctypes.c_void_p, ctypes.c_uint]

        d.csmReviveMocInPlace.restype = ctypes.c_void_p
        d.csmReviveMocInPlace.argtypes = [ctypes.c_void_p, ctypes.c_uint]

        d.csmGetSizeofModel.restype = ctypes.c_uint
        d.csmGetSizeofModel.argtypes = [ctypes.c_void_p]

        d.csmInitializeModelInPlace.restype = ctypes.c_void_p
        d.csmInitializeModelInPlace.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint]

        d.csmUpdateModel.restype = None
        d.csmUpdateModel.argtypes = [ctypes.c_void_p]

        d.csmReadCanvasInfo.restype = None
        d.csmReadCanvasInfo.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(csmVector2),
            ctypes.POINTER(csmVector2),
            ctypes.POINTER(ctypes.c_float),
        ]

        d.csmGetParameterCount.restype = ctypes.c_int
        d.csmGetParameterCount.argtypes = [ctypes.c_void_p]
        d.csmGetParameterIds.restype = ctypes.POINTER(ctypes.c_char_p)
        d.csmGetParameterIds.argtypes = [ctypes.c_void_p]
        d.csmGetParameterValues.restype = ctypes.POINTER(ctypes.c_float)
        d.csmGetParameterValues.argtypes = [ctypes.c_void_p]
        d.csmGetParameterDefaultValues.restype = ctypes.POINTER(ctypes.c_float)
        d.csmGetParameterDefaultValues.argtypes = [ctypes.c_void_p]
        d.csmGetParameterMinimumValues.restype = ctypes.POINTER(ctypes.c_float)
        d.csmGetParameterMinimumValues.argtypes = [ctypes.c_void_p]
        d.csmGetParameterMaximumValues.restype = ctypes.POINTER(ctypes.c_float)
        d.csmGetParameterMaximumValues.argtypes = [ctypes.c_void_p]

        d.csmGetPartCount.restype = ctypes.c_int
        d.csmGetPartCount.argtypes = [ctypes.c_void_p]
        d.csmGetPartIds.restype = ctypes.POINTER(ctypes.c_char_p)
        d.csmGetPartIds.argtypes = [ctypes.c_void_p]
        d.csmGetPartOpacities.restype = ctypes.POINTER(ctypes.c_float)
        d.csmGetPartOpacities.argtypes = [ctypes.c_void_p]
        d.csmGetPartParentPartIndices.restype = ctypes.POINTER(ctypes.c_int)
        d.csmGetPartParentPartIndices.argtypes = [ctypes.c_void_p]

        d.csmGetDrawableCount.restype = ctypes.c_int
        d.csmGetDrawableCount.argtypes = [ctypes.c_void_p]
        d.csmGetDrawableIds.restype = ctypes.POINTER(ctypes.c_char_p)
        d.csmGetDrawableIds.argtypes = [ctypes.c_void_p]
        d.csmGetDrawableConstantFlags.restype = ctypes.POINTER(ctypes.c_ubyte)
        d.csmGetDrawableConstantFlags.argtypes = [ctypes.c_void_p]
        d.csmGetDrawableDynamicFlags.restype = ctypes.POINTER(ctypes.c_ubyte)
        d.csmGetDrawableDynamicFlags.argtypes = [ctypes.c_void_p]
        d.csmGetDrawableTextureIndices.restype = ctypes.POINTER(ctypes.c_int)
        d.csmGetDrawableTextureIndices.argtypes = [ctypes.c_void_p]
        d.csmGetDrawableDrawOrders.restype = ctypes.POINTER(ctypes.c_int)
        d.csmGetDrawableDrawOrders.argtypes = [ctypes.c_void_p]
        d.csmGetRenderOrders.restype = ctypes.POINTER(ctypes.c_int)
        d.csmGetRenderOrders.argtypes = [ctypes.c_void_p]
        d.csmGetDrawableOpacities.restype = ctypes.POINTER(ctypes.c_float)
        d.csmGetDrawableOpacities.argtypes = [ctypes.c_void_p]
        d.csmGetDrawableMaskCounts.restype = ctypes.POINTER(ctypes.c_int)
        d.csmGetDrawableMaskCounts.argtypes = [ctypes.c_void_p]
        d.csmGetDrawableMasks.restype = ctypes.POINTER(ctypes.POINTER(ctypes.c_int))
        d.csmGetDrawableMasks.argtypes = [ctypes.c_void_p]
        d.csmGetDrawableVertexCounts.restype = ctypes.POINTER(ctypes.c_int)
        d.csmGetDrawableVertexCounts.argtypes = [ctypes.c_void_p]
        d.csmGetDrawableVertexPositions.restype = ctypes.POINTER(ctypes.POINTER(csmVector2))
        d.csmGetDrawableVertexPositions.argtypes = [ctypes.c_void_p]
        d.csmGetDrawableVertexUvs.restype = ctypes.POINTER(ctypes.POINTER(csmVector2))
        d.csmGetDrawableVertexUvs.argtypes = [ctypes.c_void_p]
        d.csmGetDrawableIndexCounts.restype = ctypes.POINTER(ctypes.c_int)
        d.csmGetDrawableIndexCounts.argtypes = [ctypes.c_void_p]
        d.csmGetDrawableIndices.restype = ctypes.POINTER(ctypes.POINTER(ctypes.c_ushort))
        d.csmGetDrawableIndices.argtypes = [ctypes.c_void_p]
        d.csmGetDrawableParentPartIndices.restype = ctypes.POINTER(ctypes.c_int)
        d.csmGetDrawableParentPartIndices.argtypes = [ctypes.c_void_p]

    def get_version(self) -> int:
        return self._dll.csmGetVersion()
