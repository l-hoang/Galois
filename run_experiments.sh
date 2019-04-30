BENCHMARKS=( "sssp/sssp" "betweennesscentrality/bc-level" "connectedcomponents/connectedcomponents" "kcore/kcore" "pagerank/pagerank-pull" )
#BENCHMARKS=( "sssp/sssp" )
#INPUTS=( "rmat15" "livejournal" "road-europe" "friendster" "rmat28" "kron30" )
INPUTS=( "rmat15" )
GRAPHTYPES=( "buffered" "od" "direct" "filegraph" "mmap" "offline" )
INIT_FLAGS=" -t=56 "
execdir=`pwd`

for benchmark in "${BENCHMARKS[@]}"; do
  for input in "${INPUTS[@]}"; do
    FLAGS=$INIT_FLAGS
    inputdir="/workspace/aos-inputs"
    ext="gr"

    if [[ ${benchmark} == *"connected"* || ${benchmark} == *"kcore"* ]]; then
      inputdir="/workspace/aos-inputs/symmetric/"
      ext="sgr"
      FLAGS+=" -symmetricGraph"
    elif [[ ${benchmark} == *"pagerank"* ]]; then
      inputdir="/workspace/aos-inputs/transpose/"
      ext="tgr"
    fi

    if [[ ${benchmark} == *"kcore"* ]]; then
      FLAGS+=" -kcore=100"
    fi

    if [[ ${benchmark} == *"bc-level"* ]]; then
      FLAGS+=" -singleSource"
    fi

    # TODO source nodes for bc, sssp
    finalinput=${inputdir}/${input}.${ext}
    bettername=`echo "$benchmark" | tr / _`

    # warmup run
    for graphtype in "${GRAPHTYPES[@]}"; do
      echo "${execdir}/${benchmark}_${graphtype} ${finalinput} ${FLAGS}"
      ${execdir}/${benchmark}_${graphtype} ${finalinput} ${FLAGS}
    done

    # assumes an outputs directory exists in current working directory
    #for i in {1..3}; do
    #  for graphtype in "${GRAPHTYPES[@]}"; do
    #    statfilename="outputs/${bettername}_${graphtype}_${input}_run${i}"
    #    ${execdir}/${benchmark}_${graphtype} ${finalinput} ${FLAGS} -statFile=${statfilename}.stats |& tee ${statfilename}.out
    #  done
    #done
  done
done
