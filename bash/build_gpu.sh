#!/bin/bash


touch build-$1.pbs
echo "#!/bin/bash" >> build-$1.pbs
echo "#PBS -P DATA3888" >> build-$1.pbs
echo "#PBS -l select=1:ncpus=4:ngpus=1:mem=16GB" >> build-$1.pbs
echo "#PBS -l walltime=48:00:00" >> build-$1.pbs
echo "#PBS -N $1" >> build-$1.pbs
echo "cd /project/DATA3888/gatsby/cueva" >> build-$1.pbs
echo "module load python/3.8.2 magma/2.5.3" >> build-$1.pbs


savedir=""

for ((i = 0; i < $#; i++)); do
  if [ "${!i}" == "-c" ]; then
    j=$((i+1))
    savedir="${!j}"
    break
  fi
  
  if [ "${!i}" == "-savedir" ]; then
    j=$((i+1))
    savedir="${!j}"
    break
  fi
done

if [ ! -d "$savedir" ]; then
  mkdir -p "$savedir"
fi

if [ -n "$savedir" ]; then
  echo "python3 -m build $@ > $savedir/build-$1.out" >> build-$1.pbs
else
  echo "python3 -m build $@ > build-$1.out" >> build-$1.pbs
fi

qsub build-$1.pbs
rm build-$1.pbs


