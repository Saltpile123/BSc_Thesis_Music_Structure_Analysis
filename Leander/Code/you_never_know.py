#np_features = [np.array(test_song['features'][feat])[:np_labels_one.shape[0],:] for feat in features]
# np_features: list(num_features * (842, feature_length))

# def print_neurons(sorted_results, top_count):
#     print('\nNeurons:')
#     print('128:', len([res for res in sorted_results if res.neurons == 128]))
#     print('256:', len([res for res in sorted_results if res.neurons == 256]))
#     print('512:', len([res for res in sorted_results if res.neurons == 512]))

#     print('Top 10 percent:')
#     print('128:', len([res for res in sorted_results[:top_count] if res.neurons == 128]))
#     print('256:', len([res for res in sorted_results[:top_count] if res.neurons == 256]))
#     print('512:', len([res for res in sorted_results[:top_count] if res.neurons == 512]))


# def print_drop_out(sorted_results, top_count):
#     print('\nDrop Out:')
#     print('0.1:', len([res for res in sorted_results if res.drop_out == 0.1]))
#     print('0.25:', len([res for res in sorted_results if res.drop_out == 0.25]))
#     print('0.5:', len([res for res in sorted_results if res.drop_out == 0.5]))

#     print('Top 10 percent:')
#     print('0.1:', len([res for res in sorted_results[:top_count] if res.drop_out == 0.1]))
#     print('0.25:', len([res for res in sorted_results[:top_count] if res.drop_out == 0.25]))
#     print('0.5:', len([res for res in sorted_results[:top_count] if res.drop_out == 0.5]))


# def print_pooling(sorted_results, top_count):
#     print('\nPooling:')
#     print('none:', len([res for res in sorted_results if res.pooling == '']))
#     print('max:', len([res for res in sorted_results if res.pooling == 'max']))
#     print('average:', len([res for res in sorted_results if res.pooling == 'average']))

#     print('Top 10 percent:')
#     print('none:', len([res for res in sorted_results[:top_count] if res.pooling == '']))
#     print('max:', len([res for res in sorted_results[:top_count] if res.pooling == 'max']))
#     print('average:', len([res for res in sorted_results[:top_count] if res.pooling == 'average']))


# def print_epochs(sorted_results, top_count):
#     print('\nEpochs:')
#     print('10:', len([res for res in sorted_results if res.epochs == 10]))
#     print('50:', len([res for res in sorted_results if res.epochs == 50]))
#     print('100:', len([res for res in sorted_results if res.epochs == 100]))

#     print('Top 10 percent:')
#     print('10:', len([res for res in sorted_results[:top_count] if res.epochs == 10]))
#     print('50:', len([res for res in sorted_results[:top_count] if res.epochs == 50]))
#     print('100:', len([res for res in sorted_results[:top_count] if res.epochs == 100]))


# def print_batch_size(sorted_results, top_count):
#     print('\nBatch Size:')
#     print('10:', len([res for res in sorted_results if res.batch_size == 10]))
#     print('50:', len([res for res in sorted_results if res.batch_size == 50]))
#     print('100:', len([res for res in sorted_results if res.batch_size == 100]))

#     print('Top 10 percent:')
#     print('10:', len([res for res in sorted_results[:top_count] if res.batch_size == 10]))
#     print('50:', len([res for res in sorted_results[:top_count] if res.batch_size == 50]))
#     print('100:', len([res for res in sorted_results[:top_count] if res.batch_size == 100]))


# def print_feature_combos(sorted_results, top_count):
#     print('\nFeature Combos:')
#     print('all', len([res for res in sorted_results if len(res.features) > 3]))
#     print('c+t', len([res for res in sorted_results if res.features == ['cqt', 'tempogram']]))
#     print('c+m', len([res for res in sorted_results if res.features == ['cqt', 'mfcc']]))
#     print('cmt', len([res for res in sorted_results if res.features == ['cqt', 'mfcc', 'tempogram']]))

#     print('Top 10 percent:')
#     print('all', len([res for res in sorted_results[:top_count] if len(res.features) > 3]))
#     print('c+t', len([res for res in sorted_results[:top_count] if res.features == ['cqt', 'tempogram']]))
#     print('c+m', len([res for res in sorted_results[:top_count] if res.features == ['cqt', 'mfcc']]))
#     print('cmt', len([res for res in sorted_results[:top_count] if res.features == ['cqt', 'mfcc', 'tempogram']]))