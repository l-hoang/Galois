BENCHMARKS=( "sssp" "kcore" )
INPUTS=( "rmat20" "rmat22" "rmat23" "rmat25" )
GRAPHTYPES=( "buffered" "od" "direct" "filegraph" "mmap" "offline" "async")
INIT_FLAGS=" -t=1 "
execdir=`pwd`"/build/lonestar/"
MAX_ITER=3

for benchmark in "${BENCHMARKS[@]}"; do
  for input in "${INPUTS[@]}"; do
    echo $benchmark":"$input >> outputs/${benchmark}_summarize
    FLAGS=$INIT_FLAGS
    inputdir="/net/ohm/export/iss/dist-inputs"
    ext="gr"

    if [[ ${benchmark} == *"connected"* || ${benchmark} == *"kcore"* ]]; then
      inputdir="/net/ohm/export/iss/dist-inputs/symmetric/"
      ext="sgr"
      FLAGS+=" -symmetricGraph"
    elif [[ ${benchmark} == *"pagerank"* ]]; then
      inputdir="/net/ohm/export/iss/dist-inputs/transpose/"
      ext="tgr"
    fi

    if [[ ${benchmark} == *"kcore"* ]]; then
      FLAGS+=" -kcore=100"
    fi

    if [[ ${benchmark} == *"bc-level"* ]]; then
      FLAGS+=" -singleSource"
    fi

    if [[ ${benchmark} == *"sssp"* ]]; then
      FLAGS+=" -startNode="`cat ${inputdir}/${input}.source`
    fi

    # TODO source nodes for bc, sssp
    finalinput=${inputdir}/${input}.${ext}
    bettername=`echo "$benchmark" | tr / _`

    # warmup run
    for graphtype in "${GRAPHTYPES[@]}"; do
      echo "${execdir}/${benchmark}/${benchmark}_${graphtype} ${finalinput} ${FLAGS}"
      ${execdir}/${benchmark}/${benchmark}_${graphtype} ${finalinput} ${FLAGS}
    done

    # warmup run
    COMPL_TIME=0.0
    echo $MAX_ITER;
    for graphtype in "${GRAPHTYPES[@]}"; do
        for ((i=1;i<=$MAX_ITER;i++)) do
          echo "${execdir}/${benchmark}/${benchmark}_${graphtype} ${finalinput} ${FLAGS}"
          ${execdir}/${benchmark}/${benchmark}_${graphtype} ${finalinput} ${FLAGS} >> outputs/${benchmark}
          MED_TIME=`cat ${execdir}/${benchmark}/${benchmark}_time`
          COMPL_TIME=`bc -l <<< "scale=7; $COMPL_TIME+$MED_TIME"`
        done
        COMPL_TIME=`bc -l <<< "scale=7; $COMPL_TIME/$MAX_ITER"`;
        echo ${graphtype}","${COMPL_TIME} >> outputs/${benchmark}_summarize
    done
  done
done
