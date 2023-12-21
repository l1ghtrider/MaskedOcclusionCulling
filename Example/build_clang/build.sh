echo building --------------------------------------------------------------------- MaaskedOcclusionCulling.cpp
clang++ -msse4.1 -c ../../MaskedOcclusionCulling.cpp -o moc_sse4.o -std=c++11 > log.txt
echo building --------------------------------------------------------------------- MaskedOcclusionCullingAVX2.cpp
clang++ -mavx2 -mfma -c ../../MaskedOcclusionCullingAVX2.cpp -o moc_avx2.o -std=c++11 >> log.txt
echo building --------------------------------------------------------------------- MaskedOcclusionCullingAVX512.cpp
clang++ -mavx512f -mfma -march=skylake-avx512 -c ../../MaskedOcclusionCullingAVX512.cpp -o moc_avx512.o -std=c++11 >> log.txt
echo building --------------------------------------------------------------------- ExampleMain.cpp
clang++ -c ../ExampleMain.cpp -o main.o -std=c++11
echo building --------------------------------------------------------------------- main.out
clang++ main.o moc_sse4.o moc_avx2.o moc_avx512.o -o main.out -lstdc++ >> log.txt
echo executing --------------------------------------------------------------------
./main.out