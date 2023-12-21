echo building --------------------------------------------------------------------- MaaskedOcclusionCulling.cpp
aarch64-linux-gnu-g++ -c ../../MaskedOcclusionCulling.cpp -o moc_base.o -std=c++11 > log_arm.txt
echo building --------------------------------------------------------------------- MaskedOcclusionCullingNeon.cpp
aarch64-linux-gnu-g++ -c ../../MaskedOcclusionCullingNeon.cpp -o moc_neon.o -std=c++11 >> log_arm.txt
echo building --------------------------------------------------------------------- ExampleMain.cpp
aarch64-linux-gnu-g++ -c ../ExampleMain.cpp -o main.o -std=c++11 >> log_arm.txt
echo building --------------------------------------------------------------------- main_arm.out
aarch64-linux-gnu-g++ main.o moc_base.o moc_neon.o -o main_arm.out --static -std=c++11 >> log_arm.txt
echo executing --------------------------------------------------------------------
./main_arm.out