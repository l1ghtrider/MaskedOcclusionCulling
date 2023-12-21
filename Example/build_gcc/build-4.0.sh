echo building --------------------------------------------------------------------- MaaskedOcclusionCulling.cpp
g++-4.0 -msse4.1 -c ../../MaskedOcclusionCulling.cpp -o moc_sse4.o -std=c++11 > log40.txt
echo building --------------------------------------------------------------------- MaskedOcclusionCullingAVX2.cpp
g++-4.0 -mavx2 -mfma -c ../../MaskedOcclusionCullingAVX2.cpp -o moc_avx2.o -std=c++11 >> log40.txt
echo building --------------------------------------------------------------------- MaskedOcclusionCullingAVX512.cpp
g++-4.0 -mavx512f -mfma -march=skylake-avx512 -c ../../MaskedOcclusionCullingAVX512.cpp -o moc_avx512.o -std=c++11 >> log40.txt
echo building --------------------------------------------------------------------- ExampleMain.cpp
g++-4.0 -c ../ExampleMain.cpp -o main.o -std=c++11
echo building --------------------------------------------------------------------- main40.out
g++-4.0 main.o moc_sse4.o moc_avx2.o moc_avx512.o -o  main40.out -std=c++11 >> log40.txt
echo executing --------------------------------------------------------------------
./main40.out
