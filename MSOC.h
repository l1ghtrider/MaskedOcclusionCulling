#pragma once
#include "Runtime/GameCode/Behaviour.h"

class MSOC {
    REGISTER_CLASS_TRAITS(kTypeNoFlags);

public:
    static void MOCCreateInstance();
    static void MOCSetResolution(int width, int height);
    static void MOCSetNearClip(float n1earClip);
    static void MOCClearBuffer();
    static void MOCDestroyInstance();
    static void MOCRenderTriangles(void* triVerts, void* triIndices
        , int nTris, const float* modelToClipMatrix, int bfWinding = 2, bool unityLayout = false);
    static void MOCRenderTrianglesSort(void* triVerts, void* triIndices
        , int nTris, const float* modelToClipMatrix, int bfWinding = 2, bool unityLayout = false);
    static void MOCRenderFlush();
    static void MOCSetQuickMask(bool enabled);
    static bool MOCTestTriangles(void* triVerts, void* triIndices
        , int nTris, const float* modelToClipMatrix, int bfWinding = 2, bool unityLayout = false);
    static bool MOCTestRect(float xmin, float ymin, float xmax, float ymax, float wmin);
    static void MOCRenderRect(float xmin, float ymin, float xmax, float ymax, float wmin);
    static void MOCGetInfo(int& mode, int& tested, int& culled, int& quickMask);
    static void MOCDrawImage();
    static void MOCRecord(bool vis);
};

BIND_MANAGED_TYPE_NAME(MSOC, UnityEngine_MSOC);
