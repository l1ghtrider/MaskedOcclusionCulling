#if USE_SOC
#include "UnityPrefix.h"
#include "MSOC.h"

#include "Runtime/Serialize/TransferFunctions/SerializeTransfer.h"

#include "Runtime/BaseClasses/IsPlaying.h"
#include "Runtime/Transform/Transform.h"

#ifndef _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS
#endif
#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <algorithm>
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#else
#ifdef _WIN32
#include <intrin.h>
#else
#include <immintrin.h>
#endif
#endif
#include "MaskedOcclusionCulling.h"
#if ENABLE_SCRIPTING_API_THREAD_AND_SERIALIZATION_CHECK
#undef ENABLE_SCRIPTING_API_THREAD_AND_SERIALIZATION_CHECK
#endif

////////////////////////////////////////////////////////////////////////////////////////
// Image utility functions, minimal BMP writer and depth buffer tone mapping
////////////////////////////////////////////////////////////////////////////////////////

static void WriteBMP(const char* filename, const unsigned char* data, int w, int h)
{
    short header[] = { 0x4D42, 0, 0, 0, 0, 26, 0, 12, 0, (short)w, (short)h, 1, 24 };
    FILE* f = fopen(filename, "wb");
    fwrite(header, 1, sizeof(header), f);
#if USE_D3D == 1
    // Flip image because Y axis of Direct3D points in the opposite direction of bmp. If the library 
    // is configured for OpenGL (USE_D3D 0) then the Y axes would match and this wouldn't be required.
    for (int y = 0; y < h; ++y)
        fwrite(&data[(h - y - 1) * w * 3], 1, w * 3, f);
#else
    fwrite(data, 1, w * h * 3, f);
#endif
    fclose(f);
}

static void TonemapDepth(float* depth, unsigned char* image, int w, int h)
{
    // Find min/max w coordinate (discard cleared pixels)
    float minW = FLT_MAX, maxW = 0.0f;
    for (int i = 0; i < w * h; ++i)
    {
        if (depth[i] > 0.0f)
        {
            minW = std::min(minW, depth[i]);
            maxW = std::max(maxW, depth[i]);
        }
    }

    // Tonemap depth values
    for (int i = 0; i < w * h; ++i)
    {
        int intensity = 0;
        if (depth[i] > 0)
            intensity = (unsigned char)(127.0 * (depth[i] - minW) / (maxW - minW) + 128.0);

        image[i * 3 + 0] = intensity;
        image[i * 3 + 1] = intensity;
        image[i * 3 + 2] = intensity;
    }
}

typedef MaskedOcclusionCulling::BackfaceWinding BackfaceWinding;
typedef MaskedOcclusionCulling::VertexLayout  VertexLayout;

static MaskedOcclusionCulling* s_moc = nullptr;
static VertexLayout s_vertexLayout = VertexLayout(12, 4, 8);
static int s_width = 1920;
static int s_height = 1080;

void MSOC::MOCCreateInstance()
{
    MOCDestroyInstance();
    s_moc = MaskedOcclusionCulling::Create();
    if (s_moc != nullptr)
    {
        MaskedOcclusionCulling::Implementation implementation = s_moc->GetImplementation();
        switch (implementation) {
        case MaskedOcclusionCulling::SSE2: LogString("Using SSE2 version\n"); break;
        case MaskedOcclusionCulling::SSE41: LogString("Using SSE41 version\n"); break;
        case MaskedOcclusionCulling::AVX2: LogString("Using AVX2 version\n"); break;
        case MaskedOcclusionCulling::AVX512: LogString("Using AVX-512 version\n"); break;
        case MaskedOcclusionCulling::NEON: LogString("Using NEON version\n"); break;
        default: break;
        }
    }
    else
    {
        LogString("MOC Creating Failed!\n");
    }
}

void MSOC::MOCSetResolution(int width, int height)
{
    if (s_moc == nullptr) return;
    s_width = width;
    s_height = height;
    s_moc->SetResolution(width, height);
}

void MSOC::MOCSetNearClip(float nearClip)
{
    if (s_moc == nullptr) return;
    s_moc->SetNearClipPlane(nearClip);
}

void MSOC::MOCClearBuffer()
{
    if (s_moc == nullptr) return;
    s_moc->ClearBuffer();
}

void MSOC::MOCDestroyInstance()
{
    if (s_moc != nullptr)
    {
        MaskedOcclusionCulling::Destroy(s_moc);
        s_moc = nullptr;
    }
}

void MSOC::MOCRenderTriangles(void* triVerts, void* triIndices
    , int nTris, const float* modelToClipMatrix, int bfWinding, bool unityLayout)
{
    if (s_moc == nullptr) return;
    if (unityLayout)
    {
        s_moc->RenderTriangles((float*)triVerts, (const unsigned int*)triIndices, nTris, modelToClipMatrix
            , (BackfaceWinding)bfWinding, MaskedOcclusionCulling::CLIP_PLANE_ALL, s_vertexLayout);
    }
    else
    {
        s_moc->RenderTriangles((float*)triVerts, (const unsigned int*)triIndices, nTris, modelToClipMatrix
            , (BackfaceWinding)bfWinding, MaskedOcclusionCulling::CLIP_PLANE_ALL);
    }
}

void MSOC::MOCRenderTrianglesSort(void* triVerts, void* triIndices
    , int nTris, const float* modelToClipMatrix, int bfWinding, bool unityLayout)
{
    if (s_moc == nullptr) return;
    if (unityLayout)
    {
        s_moc->RenderTrianglesSort((float*)triVerts, (const unsigned int*)triIndices, nTris, modelToClipMatrix
            , (BackfaceWinding)bfWinding, MaskedOcclusionCulling::CLIP_PLANE_ALL, s_vertexLayout);
    }
    else
    {
        s_moc->RenderTrianglesSort((float*)triVerts, (const unsigned int*)triIndices, nTris, modelToClipMatrix
            , (BackfaceWinding)bfWinding, MaskedOcclusionCulling::CLIP_PLANE_ALL);
    }
}

void MSOC::MOCRenderFlush()
{
    if (s_moc == nullptr) return;
    s_moc->RenderFlush();
}

void MSOC::MOCSetQuickMask(bool enabled)
{
    if (s_moc == nullptr) return;
    s_moc->quickMask = enabled;
}

bool MSOC::MOCTestTriangles(void* triVerts, void* triIndices
    , int nTris, const float* modelToClipMatrix, int bfWinding, bool unityLayout)
{
    if (s_moc == nullptr) return true;
    MaskedOcclusionCulling::CullingResult result;
    if (unityLayout)
    {
        result = s_moc->TestTriangles((float*)triVerts, (const unsigned int*)triIndices, nTris, modelToClipMatrix
            , (BackfaceWinding)bfWinding, MaskedOcclusionCulling::CLIP_PLANE_SIDES, s_vertexLayout);
    }
    else
    {
        result = s_moc->TestTriangles((float*)triVerts, (const unsigned int*)triIndices, nTris, modelToClipMatrix
            , (BackfaceWinding)bfWinding, MaskedOcclusionCulling::CLIP_PLANE_SIDES);
    }
    return result == MaskedOcclusionCulling::VISIBLE;
}

bool MSOC::MOCTestRect(float xmin, float ymin, float xmax, float ymax, float wmin)
{
    if (s_moc == nullptr) return true;
    auto result = s_moc->TestRect(xmin, ymin, xmax, ymax, wmin);
    return result == MaskedOcclusionCulling::VISIBLE;
}

void MSOC::MOCRenderRect(float xmin, float ymin, float xmax, float ymax, float wmin)
{
    // A triangle that intersects the view frustum
    struct ClipspaceVertex { float x, y, z, w; };
    xmin *= wmin; ymin *= wmin; xmax *= wmin; ymax *= wmin;
    ClipspaceVertex triVerts1[] = { { xmin, ymin, wmin, wmin }, { xmax, ymin, wmin, wmin }, { xmin, ymax, wmin, wmin }, { xmax, ymax, wmin, wmin } };
    unsigned int triIndices1[] = { 0, 2, 1, 1, 2, 3 };
    // Render the triangle
    s_moc->RenderTriangles((float*)triVerts1, triIndices1, 2);
}

void MSOC::MOCDrawImage()
{
    if (s_moc == nullptr) return;
    // Compute a per pixel depth buffer from the hierarchical depth buffer, used for visualization.
    float* perPixelZBuffer = new float[s_width * s_height];
    s_moc->ComputePixelDepthBuffer(perPixelZBuffer, false);

    // Tonemap the image
    unsigned char* image = new unsigned char[s_width * s_height * 3];
    TonemapDepth(perPixelZBuffer, image, s_width, s_height);
    WriteBMP("image.bmp", image, s_width, s_height);
    delete[] image;
}

void MSOC::MOCRecord(bool vis)
{
#if MOC_DEBUG != 0
    if (s_moc == nullptr) return;
    s_moc->tested++;
    s_moc->culled += vis ? 1 : 0;
#endif
}

void MSOC::MOCGetInfo(int& mode, int& tested, int& culled, int& quickMask)
{
#if MOC_DEBUG != 0
    if (s_moc == nullptr)
    {
        mode = -1;
        tested = -1;
        culled = -1;
        quickMask = 1;
    }
    else
    {
        mode = (int)s_moc->GetImplementation();
        tested = s_moc->tested;
        culled = s_moc->culled;
        quickMask = s_moc->GetRealQuickMask();
    }
#else
    mode = -1;
    tested = -1;
    culled = -1;
    quickMask = 1;
#endif
}

#endif
