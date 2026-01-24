#!/bin/bash

touch build-$1.sh
echo "#!/bin/bash" >> build-$1.sh
echo "#SBATCH -J build-$1" >> build-$1.sh
echo "#SBATCH -p gpu" >> build-$1.sh
echo "#SBATCH -N 1" >> build-$1.sh
echo "#SBATCH -n 4" >> build-$1.sh
echo "#SBATCH --mem 16G" >> build-$1.sh
echo "#SBATCH --gres gpu:1" >> build-$1.sh
echo "#SBATCH -t 0-06:00" >> build-$1.sh
echo "#SBATCH --mail-type=ALL" >> build-$1.sh
echo "#SBATCH --mail-user=h1d2y6c0e0u4q1z8@gatsbyunit.slack.com" >> build-$1.sh

echo "cd /nfs/ghome/live/jlaherty/Orientation2Shelter" >> build-$1.sh
echo "source /etc/profile.d/modules.sh" >> build-$1.sh
echo "module load miniconda" >> build-$1.sh

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
  echo "conda run -p /nfs/ghome/live/jlaherty/anaconda3/envs/O2S python3 -m o2s.build $@ -wandb > $savedir/build-$1.out" >> build-$1.sh
else
  echo "conda run -p /nfs/ghome/live/jlaherty/anaconda3/envs/O2S python3 -m o2s.build $@ -wandb > build-$1.out" >> build-$1.sh
fi

sbatch build-$1.sh
rm build-$1.sh


