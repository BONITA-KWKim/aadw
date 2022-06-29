[ copy raw images ]
find . -name S_*.svs  | xargs -n1 -I{} cp {} /data/118/work/raw-v4.0
find . -name UB*.svs  | xargs -n1 -I{} cp {} /data/118/work/raw-v4.0
find . -name C20*     | xargs -n1 -I{} cp {} /data/118/work/raw-v4.0
find . -name *HE*.svs | xargs -n1 -I{} cp {} /data/118/work/raw-v4.0

[ accumulate thumbnail images ]
mkdir thumbnail
ls */thumbnail_* | xargs -n1 -I{} cp {} thumbnail/
mv thumbnail ../HE-nasnet-umap-100-thumbnail/



[hierarchical clustering]
위치: (118번서버) /data/work/thumbnails
(종류)
+-----+-----------------------------------------+---------------+-----------+---------+
| 순번 |              폴  더  명                   | Normalization |  차원 축소  | 특징 추출 |
+-----+-----------------------------------------+---------------+-----------+---------+
|  1  | EOSIN-nasnet-umap-100-thumbnail/        | EOSIN         | UMAP(100) | Nasnet  |
+-----+-----------------------------------------+---------------+-----------+---------+
|  2  | HE-nasnet-umap-100-thumbnail/           | H&E           | UMAP(100) | Nasnet  |
+-----+-----------------------------------------+---------------+-----------+---------+
|  3  | HEMATOXYLIN-nasnet-umap-100-thumbnail/  | HEMATOXYLIN   | UMAP(100) | Nasnet  |
+-----+-----------------------------------------+---------------+-----------+---------+
|  4  | vgg_thumbnail/                          | None          | None      | VGG     |
+-----+-----------------------------------------+---------------+-----------+---------+
|  5  | vgg_umap_100_thumbnail/                 | None          | UMAP(100) | VGG     |
+-----+-----------------------------------------+---------------+-----------+---------+
