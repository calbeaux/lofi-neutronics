**Building SyTH Library** `libsyTH.so`

cd ~/syth/build && rm -rf * && source ~/syth_tpls/load_env.sh && ~/syth/build_scripts/build_with_syth_env.sh ~/syth/build_scripts/build_petsc_release_omp_all.sh ~/syth -DSYTH_ENABLE_Fortran=OFF -DBUILD_SHARED_LIBS=ON && make -j

currently works with branch lofi-neutronics-coupling
