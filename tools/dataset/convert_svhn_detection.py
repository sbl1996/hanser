import h5py
import imagesize

from hhutil.io import fmt_path, save_json

root = fmt_path("/Users/hrvvi/Downloads/test")

f = h5py.File(root / "digitStruct.mat")
d = f['digitStruct']

def read_name(ref):
    x = d[ref][:].tostring().decode('UTF-16LE')
    return x

def read_bboxes(ref):
    g = d[ref]
    keys = list(g.keys())
    n = g[keys[0]].shape[0]
    results = []
    if n == 1:
        xs = []
        for k in ['left', 'top', 'width', 'height', 'label']:
            x = g[k][:][0, 0]
            xs.append(x)
        bboxes = [xs[:4]]
        labels = [xs[4]]
        return bboxes, labels
    else:
        for k in ['left', 'top', 'width', 'height', 'label']:
            xs = []
            kg = g[k][:]
            for i in range(n):
                x = d[kg[i, 0]][:][0, 0]
                xs.append(x)
            results.append(xs)
        bboxes = [[results[j][i] for j in range(4)] for i in range(n)]
        labels = [results[-1][i] for i in range(n)]
        return bboxes, labels

# annotation_id = 0
# offset = 0

annotation_id = 73257
offset = 33402
images = []
annotations = []
for i in range(d['name'].shape[0]):
    print(i)
    file_name = read_name(d['name'][i][:][0])
    image_id = offset + int(file_name[:-4])
    width, height = imagesize.get(root / file_name)
    images.append({
        'file_name': file_name,
        'height': height,
        'width': width,
        'id': image_id,
    })

    bboxes, labels = read_bboxes(d['bbox'][i][:][0])
    nb = len(labels)
    for j in range(nb):
        bbox = bboxes[j]
        annotation_id += 1
        annotations.append({
            'iscrowd': 0,
            'area': bbox[2] * bbox[3],
            'image_id': image_id,
            'bbox': bbox,
            'category_id': int(labels[j]),
            'id': annotation_id,
        })

categories = [ {'supercategory': 'digit', 'id': i+1, 'name': str(i)} for i in range(10)]
anns = {
    'images': images,
    'annotations': annotations,
    'categories': categories,
}
save_json(anns, "/Users/hrvvi/Downloads/test.json")