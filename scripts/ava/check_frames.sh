find /mnt/local-drive/ava/frames -maxdepth 1 -mindepth 1 -type d | while read dir; do
  printf "%-50.50s : " "$dir"
  find "$dir" -type f | wc -l
done
