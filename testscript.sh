rm output_*
./sssp_offline /workspace/aos-inputs/rmat15.gr -t=56 -verify
diff ground output_*
rm output_*
./sssp_filegraph /workspace/aos-inputs/rmat15.gr -t=56 -verify
diff ground output_*
rm output_*
./sssp_direct /workspace/aos-inputs/rmat15.gr -t=56 -verify
diff ground output_*
rm output_*
./sssp_buffered /workspace/aos-inputs/rmat15.gr -t=56 -verify
diff ground output_*
rm output_*
./sssp_mmap /workspace/aos-inputs/rmat15.gr -t=56 -verify
diff ground output_*
rm output_*
./sssp_od /workspace/aos-inputs/rmat15.gr -t=56 -verify
diff ground output_*
rm output_*
