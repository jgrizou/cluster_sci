# rsync -avun --itemize-changes ../cluster_sci/ ../../OneDrive\ -\ University\ of\ Glasgow/cluster_sci/ | grep '^>f' | awk '{print $2}'

# rsync -avun --stats ../cluster_sci/ ../../OneDrive\ -\ University\ of\ Glasgow/cluster_sci/

rsync -avu --progress ../cluster_sci/ ../../OneDrive\ -\ University\ of\ Glasgow/cluster_sci/
