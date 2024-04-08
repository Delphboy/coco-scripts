mkdir coco
cd coco

wget -c http://images.cocodataset.org/zips/train2014.zip
wget -c http://images.cocodataset.org/zips/val2014.zip
wget -c http://images.cocodataset.org/zips/test2014.zip

unzip train2014.zip
unzip val2014.zip
unzip test2014.zip

rm train2014.zip
rm val2014.zip
rm test2014.zip

wget https://github.com/Delphboy/karpathy-splits/raw/main/dataset_coco.json?download= -O dataset_coco.json