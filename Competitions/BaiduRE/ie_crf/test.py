import ie
import ee
from tqdm import tqdm
import json

# predicate2type[l['predicate']] = (l['subject_type'], l['object_type'])
def predict_to_file(l):
    """预测结果到文件，方便提交
    """
    spoes = []
    subs = ie.extract_spoes(l['text'])
    print("subject:",subs)
    for subject in subs:
        pk = ee.extract_arguments(l['text'],subject) 
        print(subject,"##",pk)
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
    return l
            
l = {"text": "《吸血鬼偶像》是李根旭指导的一部情景喜剧，集结了洪宗玄、金宇彬等众多年轻偶像，并由搞笑明星申东烨和实力演员金秀美参演配角，讲述了一个吸血鬼星球傻乎乎的王子和他的护卫们来到地球上，为了成为明星而孤军奋斗的故事"}   

print(predict_to_file(l))
