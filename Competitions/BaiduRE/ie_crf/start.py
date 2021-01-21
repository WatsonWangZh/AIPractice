import ie
import ee
from tqdm import tqdm
import json
import os
os.environ['CUDA_VISIABLE_DIVICE'] = '1'

# predicate2type[l['predicate']] = (l['subject_type'], l['object_type'])
def predict_to_file(in_file, out_file):
    """预测结果到文件，方便提交
    """
    fw = open(out_file, 'w', encoding='utf-8')
    with open(in_file) as fr:
        for l in tqdm(fr):
            l = json.loads(l)
            spoes = []
            subs = ie.extract_spoes(l['text'])
            for subject in subs:
                pk = ee.extract_arguments(l['text'],subject) 
                for obj,p in pk.items():
                    spoes.append({
                        'subject': subject,
                        'subject_type': ee.predicate2subtype[p.split("_")[0]],
                        'predicate': p.split("_")[0],
                        'object': {p.split("_")[1]:obj},
                        'object_type': {
                            p.split("_")[1]:ie.predicate2type[p]
                            }
                    })
            l['spo_list'] = spoes
            s = json.dumps(l, ensure_ascii=False)
            fw.write(s + '\n')
    fw.close()

predict_to_file('../data/test1_data/test1_data.json', 'ie_pred.json')
