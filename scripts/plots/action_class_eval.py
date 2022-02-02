import matplotlib.pyplot as pyplot
import json
import numpy as np

def parse_slowfast(fh):
    rets = []
    for line in fh:
        if not 'road_eval_helper.py: 240:' in line:
            continue
        json_str = line.split('road_eval_helper.py: 240:')[-1].strip().replace("'", '"') 
        rets.append(json.loads(json_str))
    return rets

def parse_acar(fh):
    read = False
    json_str = ""
    rets = []
    for line in fh:
        if read:
            json_str += line.strip().replace("'", '"') 
            if '}' in line:
                rets.append(json.loads(json_str))
                read = False
                json_str = ""
            continue
        if not 'calc_mAP.py][line: 255]' in line:
            continue
        read = True
        json_str += line.split("calc_mAP.py][line: 255][    INFO] ")[-1].strip().replace("'", '"') 
    return rets



def extract_mAP(d):
    return float(d['PascalBoxes_Precision/mAP@0.5IOU'])*100

def extract_cat_scores(d):
    return sorted([d[i]*100 for i in d if 'mAP' not in i], reverse=True)

def extract_cat_labels(d):
    sorted_items = sorted(d.items(), key=lambda x: x[1], reverse=True)
    return [i[0].split('/')[-1].strip() for i in sorted_items if 'mAP' not in i[0]]

if __name__ == '__main__':
    ffw_s = parse_slowfast(open('./scripts/plots/ffw_s.txt', 'r'))
    ffw_p = parse_slowfast(open('./scripts/plots/ffw_p.txt', 'r'))
    acar_p = parse_acar(open('./scripts/plots/acar_p.txt', 'r'))

    ffw_s_mAP = list(map(extract_mAP, ffw_s))
    ffw_p_mAP = list(map(extract_mAP, ffw_p))
    acar_p_mAP = list(map(extract_mAP, acar_p))
    pyplot.plot(range(1, len(ffw_s_mAP) + 1), ffw_s_mAP, label='SlowFast + FFW (Stratch)')
    pyplot.plot(range(1, len(ffw_p_mAP) + 1), ffw_p_mAP, label='SlowFast + FFW (Pretrain)')
    pyplot.plot(range(1, len(acar_p_mAP) + 1), acar_p_mAP, label='SlowFast + ACAR (Pretrain)')
    pyplot.xlabel('Epoch')
    pyplot.ylabel('mAP%', labelpad=20)
    pyplot.legend()
    pyplot.savefig('action_class_eval.png')

    def subcategorybar(X, labels, vals, width=0.8):
        n = len(vals)
        _X = np.arange(len(X))
        for i in range(n):
            pyplot.bar(_X - width/2. + i/float(n)*width, vals[i], 
                    width=width/float(n), align="edge", label=labels[i])   
        pyplot.xticks(_X, X, rotation=90)

    pyplot.figure()
    subcategorybar(extract_cat_labels(ffw_s[0]), ['SlowFast + FFW (Stratch)', 'SlowFast + FFW (Pretrain)', 'SlowFast + ACAR (Pretrain)'], [extract_cat_scores(max(ffw_s, key=extract_mAP)), extract_cat_scores(max(ffw_p, key=extract_mAP)), extract_cat_scores(max(acar_p, key=extract_mAP))])
    pyplot.tight_layout()
    pyplot.ylabel('mAP%', labelpad=0)
    pyplot.legend()
    pyplot.savefig('action_class_bars.png')
    exit()
    X = np.arange(4)
    fig = pyplot.figure()
    ax = fig.add_axes([0,0,1,1])
    print(extract_cat_scores(max(ffw_s, key=extract_mAP)))
    ax.bar(X + 0.00, extract_cat_scores(max(ffw_s, key=extract_mAP)), color = 'b', width = 0.25, label='SlowFast + FFW (Stratch)')
    ax.bar(X + 0.25, extract_cat_scores(max(ffw_p, key=extract_mAP)), color = 'g', width = 0.25, label='SlowFast + FFW (Pretrain)')
    ax.bar(X + 0.50, extract_cat_scores(max(ffw_p, key=extract_mAP)), color = 'r', width = 0.25, label='SlowFast + ACAR (Pretrain)')
    fig.legend()
    pyplot.savefig('action_class_bars.png')