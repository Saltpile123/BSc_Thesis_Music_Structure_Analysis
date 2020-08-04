import os, json

import paths as pa


#  0   1   2     3   4  5 6  7  8->
# cnn_128_relu_adam_0.1__10_10_cqt_cens_pcp_tonnetz_mfcc_tempogram_class_report
class Result:
    params = []
    features = []
    accuracy = {}

    def __init__(self, params: [], micro_avg: {}):
        params = params[:-2]
        self.params = params

        self.kind = params[0]
        self.neurons = int(params[1])
        self.activation = params[2]
        self.optimizer = params[3]

        self.drop_out = float(params[4])
        self.pooling = params[5]

        self.epochs = int(params[6])
        self.batch_size = int(params[7])

        features = []
        for feat in params[8:]:
            features.append(feat)
        self.features = features

        self.accuracy = micro_avg
    
    def print_res(self):
        return self.kind


def remove_predictions(pred_path):
    for file in os.listdir(pred_path):
        if "_pred" in file:
            print('removing', file)
            os.remove(pred_path + "/" + file)


def get_results(pred_path): 
    results = []
    for file in os.listdir(pred_path):
        if '_report' in file:
            #print(file)
            with open(pred_path + '/' + file, 'r') as jsong:
                report = json.load(jsong) 
            
            params = file.split('_')

            if 'micro avg' in report:
                acc = report['micro avg']['f1-score']
            elif 'accuracy' in report:
                acc = report['accuracy']
            else:
                acc = 0

            results.append(Result(params, acc))
        
    return results


def print_neurons(sorted_results, top_count):
    print('\nNeurons:')
    print('128:', len([res for res in sorted_results if res.neurons == 128]))
    print('256:', len([res for res in sorted_results if res.neurons == 256]))
    print('512:', len([res for res in sorted_results if res.neurons == 512]))

    print('Top 10 percent:')
    print('128:', len([res for res in sorted_results[:top_count] if res.neurons == 128]))
    print('256:', len([res for res in sorted_results[:top_count] if res.neurons == 256]))
    print('512:', len([res for res in sorted_results[:top_count] if res.neurons == 512]))


def print_drop_out(sorted_results, top_count):
    print('\nDrop Out:')
    print('0.1:', len([res for res in sorted_results if res.drop_out == 0.1]))
    print('0.25:', len([res for res in sorted_results if res.drop_out == 0.25]))
    print('0.5:', len([res for res in sorted_results if res.drop_out == 0.5]))

    print('Top 10 percent:')
    print('0.1:', len([res for res in sorted_results[:top_count] if res.drop_out == 0.1]))
    print('0.25:', len([res for res in sorted_results[:top_count] if res.drop_out == 0.25]))
    print('0.5:', len([res for res in sorted_results[:top_count] if res.drop_out == 0.5]))


def print_pooling(sorted_results, top_count):
    print('\nPooling:')
    print('none:', len([res for res in sorted_results if res.pooling == '']))
    print('max:', len([res for res in sorted_results if res.pooling == 'max']))
    print('average:', len([res for res in sorted_results if res.pooling == 'average']))

    print('Top 10 percent:')
    print('none:', len([res for res in sorted_results[:top_count] if res.pooling == '']))
    print('max:', len([res for res in sorted_results[:top_count] if res.pooling == 'max']))
    print('average:', len([res for res in sorted_results[:top_count] if res.pooling == 'average']))


def print_epochs(sorted_results, top_count):
    print('\nEpochs:')
    print('10:', len([res for res in sorted_results if res.epochs == 10]))
    print('50:', len([res for res in sorted_results if res.epochs == 50]))
    print('100:', len([res for res in sorted_results if res.epochs == 100]))

    print('Top 10 percent:')
    print('10:', len([res for res in sorted_results[:top_count] if res.epochs == 10]))
    print('50:', len([res for res in sorted_results[:top_count] if res.epochs == 50]))
    print('100:', len([res for res in sorted_results[:top_count] if res.epochs == 100]))


def print_batch_size(sorted_results, top_count):
    print('\nBatch Size:')
    print('10:', len([res for res in sorted_results if res.batch_size == 10]))
    print('50:', len([res for res in sorted_results if res.batch_size == 50]))
    print('100:', len([res for res in sorted_results if res.batch_size == 100]))

    print('Top 10 percent:')
    print('10:', len([res for res in sorted_results[:top_count] if res.batch_size == 10]))
    print('50:', len([res for res in sorted_results[:top_count] if res.batch_size == 50]))
    print('100:', len([res for res in sorted_results[:top_count] if res.batch_size == 100]))


def print_feature_combos(sorted_results, top_count):
    print('\nFeature Combos:')
    print('all', len([res for res in sorted_results if len(res.features) > 3]))
    print('c+t', len([res for res in sorted_results if res.features == ['cqt', 'tempogram']]))
    print('c+m', len([res for res in sorted_results if res.features == ['cqt', 'mfcc']]))
    print('cmt', len([res for res in sorted_results if res.features == ['cqt', 'mfcc', 'tempogram']]))

    print('Top 10 percent:')
    print('all', len([res for res in sorted_results[:top_count] if len(res.features) > 3]))
    print('c+t', len([res for res in sorted_results[:top_count] if res.features == ['cqt', 'tempogram']]))
    print('c+m', len([res for res in sorted_results[:top_count] if res.features == ['cqt', 'mfcc']]))
    print('cmt', len([res for res in sorted_results[:top_count] if res.features == ['cqt', 'mfcc', 'tempogram']]))


# ATM 667 models
def print_results(sorted_results, top_count):   
    print_neurons(sorted_results, top_count)
   
    print_drop_out(sorted_results, top_count)
   
    print_pooling(sorted_results, top_count)
   
    print_epochs(sorted_results, top_count)
   
    print_batch_size(sorted_results, top_count)

    print_feature_combos(sorted_results, top_count)


print('Getting results...')
res_dir = pa.reduced_results_dir
results = get_results(res_dir)

print('Sorting results...')
sorted_results = sorted(results, key= lambda x: x.accuracy, reverse=True)
#top_sorted_results = [r for r in sorted_results if r.accuracy > 0.45]

print('Total results count:', len(results))

top_count = 70
#print_results(sorted_results, top_count)
#print_feature_combos(sorted_results, top_count)

for res in sorted_results[:70]:
    print(str(round(res.accuracy, 3)) + '\t', res.params)