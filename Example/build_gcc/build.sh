echo building --------------------------------------------------------------------- MaaskedOcclusionCulling.cpp
g++ -msse4.1 -mfma -c ../../MaskedOcclusionCulling.cpp -o moc_sse4.o -std=c++11 > log.txt
echo building --------------------------------------------------------------------- MaskedOcclusionCullingAVX2.cpp
g++ -mavx2 -mfma -c ../../MaskedOcclusionCullingAVX2.cpp -o moc_avx2.o -std=c++11 >> log.txt
echo building --------------------------------------------------------------------- MaskedOcclusionCullingAVX512.cpp
g++ -mavx512f -mfma -march=skylake-avx512 -c ../../MaskedOcclusionCullingAVX512.cpp -o moc_avx512.o -std=c++11 >> log.txt
echo building --------------------------------------------------------------------- ExampleMain.cpp
g++ -c ../ExampleMain.cpp -o main.o -std=c++11 >> log.txt
echo building --------------------------------------------------------------------- main.out
g++ main.o moc_sse4.o moc_avx2.o moc_avx512.o -o main.out -std=c++11 >> log.txt
echo executing --------------------------------------------------------------------
./main.out