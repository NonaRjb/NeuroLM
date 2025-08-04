import os
import pickle
import argparse

from multiprocessing import Pool
from sklearn.utils import shuffle
import numpy as np
import mne


chan_order = ['Fp1', 'Fp2', 'AF7', 'AF3', 'AFz', 'AF4', 'AF8', 'F7', 'F5', 'F3',
				  'F1', 'F2', 'F4', 'F6', 'F8', 'FT9', 'FT7', 'FC5', 'FC3', 'FC1', 
				  'FCz', 'FC2', 'FC4', 'FC6', 'FT8', 'FT10', 'T7', 'C5', 'C3', 'C1',
				  'Cz', 'C2', 'C4', 'C6', 'T8', 'TP9', 'TP7', 'CP5', 'CP3', 'CP1', 
				  'CPz', 'CP2', 'CP4', 'CP6', 'TP8', 'TP10', 'P7', 'P5', 'P3', 'P1',
				  'Pz', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO3', 'POz', 'PO4', 'PO8',
				  'O1', 'Oz', 'O2']


def split_and_dump(train_data, test_data, save_path, sub, ch_names, times, train_img_conditions, seed):
    
    save_dir = os.path.join(save_path, 'processed')
    os.makedirs(save_dir, exist_ok=True)
    filename_test = 'preprocessed_eeg_test.npy'
    filename_train = 'preprocessed_eeg_training.npy'
    filename_val = 'preprocessed_eeg_validation.npy'

    img_metadata_train = np.load("/proj/rep-learning-robotics/users/x_nonra/NeuroLM/data/things_eeg_2/images/image_metadata_train.npy", allow_pickle=True).item()
    img_metadata_val = np.load("/proj/rep-learning-robotics/users/x_nonra/NeuroLM/data/things_eeg_2/images/image_metadata_val.npy", allow_pickle=True).item()
    img_metadata_test = np.load("/proj/rep-learning-robotics/users/x_nonra/NeuroLM/data/things_eeg_2/images/image_metadata_test.npy", allow_pickle=True).item()
    
    def save_sample_wise(data, channels, t, metadata, split):
        save_path = os.path.join(save_dir, split)
        os.makedirs(save_path, exist_ok=True)
        for i, sample in enumerate(data):
            sample_dict = {
                'X': sample,
                'y': metadata['img_files'][i],
                'ch_names': channels,
                'times': t,
            }
            basename = metadata['img_files'][i].split('.')[0]
            with open(os.path.join(save_path, f"{basename}_sub-{format(sub,'02')}.pkl"), 'wb') as f:
                pickle.dump(sample_dict, f, protocol=4)
        return
    
    merged_test = np.concatenate(test_data, axis=1)
    idx = shuffle(np.arange(0, merged_test.shape[1]), random_state=seed)
    merged_test = merged_test[:, idx]
    print('Shape of merged test data:', merged_test.shape)
    
    all_train = np.concatenate(train_data, axis=0)
    img_cond = np.concatenate(train_img_conditions, axis=0)

    merged_train = np.zeros((len(np.unique(img_cond)), all_train.shape[1]*2,
		all_train.shape[2], all_train.shape[3]))

    for i in range(len(np.unique(img_cond))):
		# Find the indices of the selected category
	    idx = np.where(img_cond == i+1)[0]
	    for r in range(len(idx)):
	        if r == 0:
	            ordered_data = all_train[idx[r]]
	        else:
	            ordered_data = np.append(ordered_data, all_train[idx[r]], 0)
	    merged_train[i] = ordered_data
	# Shuffle the repetitions of different sessions
    idx = shuffle(np.arange(0, merged_train.shape[1]), random_state=seed)
    merged_train = merged_train[:,idx]
    print('Shape of merged train data:', merged_train.shape)    
    
    merged_val = merged_train[img_metadata_val['indices']]
    merged_train = merged_train[img_metadata_train['indices']]    
    print('Shape of merged train data after split:', merged_train.shape)
    print('Shape of merged val data after split:', merged_val.shape)
    # Insert the data into a dictionary
    test_dict = {
		'preprocessed_eeg_data': merged_test,
		'ch_names': ch_names,
		'times': times
	}
    train_dict = {
    	'preprocessed_eeg_data': merged_train,
    	'ch_names': ch_names,
    	'times': times
    } 
    val_dict = {
    	'preprocessed_eeg_data': merged_val,
    	'ch_names': ch_names,
    	'times': times
    } 

    save_sample_wise(merged_test, ch_names, times, img_metadata_test, split='test')
    save_sample_wise(merged_train, ch_names, times, img_metadata_train, split='train')
    save_sample_wise(merged_val, ch_names, times, img_metadata_val, split='val') 
    
    os.makedirs(os.path.join(save_dir, 'sub-'+format(sub,'02')), exist_ok=True)
    
    with open(os.path.join(save_dir, 'sub-'+format(sub,'02'), filename_test), 'wb') as f:
        pickle.dump(test_dict, f, protocol=4)
    with open(os.path.join(save_dir, 'sub-'+format(sub,'02'), filename_train), 'wb') as f:
        pickle.dump(train_dict, f, protocol=4)
    with open(os.path.join(save_dir, 'sub-'+format(sub,'02'), filename_val), 'wb') as f:
        pickle.dump(val_dict, f, protocol=4)

    return


def epoch_data(root, sub, n_ses, data_part, seed):
    epoched_data = []
    img_conditions = []
    for s in range(n_ses):
        ### Load the EEG data and convert it to MNE raw format ###
        eeg_dir = os.path.join('raw_eeg', 'sub-'+
        	format(sub,'02'), 'ses-'+format(s+1,'02'), 'raw_eeg_'+
        	data_part+'.npy')
        eeg_data = np.load(os.path.join(root, eeg_dir),
        	allow_pickle=True).item()
        ch_names = eeg_data['ch_names']
        sfreq = eeg_data['sfreq']
        ch_types = eeg_data['ch_types']
        eeg_data = eeg_data['raw_eeg_data']
        # Convert to MNE raw format
        info = mne.create_info(ch_names, sfreq, ch_types)
        raw = mne.io.RawArray(eeg_data, info)        
        del eeg_data        
        ### Get events, drop unused channels and reject target trials ###
        events = mne.find_events(raw, stim_channel='stim')
        # # Select only occipital (O) and posterior (P) channels
        # chan_idx = np.asarray(mne.pick_channels_regexp(raw.info['ch_names'],
        # 	'^O *|^P *'))
        # new_chans = [raw.info['ch_names'][c] for c in chan_idx]
        # raw.pick_channels(new_chans)
        # * chose all channels
        raw.pick_channels(chan_order, ordered=True)
        raw.filter(l_freq=0.1, h_freq=75.0)
        raw.notch_filter(50.0)
        # Reject the target trials (event 99999)
        idx_target = np.where(events[:,2] == 99999)[0]
        events = np.delete(events, idx_target, 0)
        ### Epoching, baseline correction and resampling ###
        # * [0, 1.0]
        epochs = mne.Epochs(raw, events, tmin=-.2, tmax=1.0, baseline=(None,0),
        	preload=True)
        # epochs = mne.Epochs(raw, events, tmin=-.2, tmax=.8, baseline=(None,0),
        # 	preload=True)
        del raw
        epochs.resample(sfreq=200, n_jobs=5)
        ch_names = epochs.info['ch_names']
        ch_names = [ch.upper() for ch in ch_names]
        times = epochs.times        
        ### Sort the data ###
        data = epochs.get_data(units='uV')
        events = epochs.events[:,2]
        print('Number of EEG trials:', len(events))
        img_cond = np.unique(events)
        del epochs
        # Select only a maximum number of EEG repetitions
        print('data_part:', data_part)
        print('eeg_dir:', eeg_dir)
        if data_part == 'test':
        	max_rep = 20
        else:
        	max_rep = 2
        # Sorted data matrix of shape:
        # Image conditions × EEG repetitions × EEG channels × EEG time points
        sorted_data = np.zeros((len(img_cond),max_rep,data.shape[1],
        	data.shape[2]))
        for i in range(len(img_cond)):
            # Find the indices of the selected image condition
            idx = np.where(events == img_cond[i])[0]
        	# Randomly select only the max number of EEG repetitions
            idx = shuffle(idx, random_state=seed, n_samples=max_rep)
            sorted_data[i] = data[idx]
        del data
        epoched_data.append(sorted_data[:, :, :, 40:]) # Remove the first 40 time points which is the 0.2 s baseline activity
        img_conditions.append(img_cond)
        del sorted_data

	### Output ###
    return epoched_data, img_conditions, ch_names, times


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare THINGS EEG2 dataset")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the raw EEG data")
    parser.add_argument("--dump_folder", type=str, required=True, help="Folder to dump processed data")
    parser.add_argument('--sub', default=10, type=int)
    parser.add_argument('--n_ses', default=4, type=int)
    parser.add_argument('--sfreq', default=200, type=int)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    dump_folder = args.dump_folder
    if not os.path.exists(dump_folder):
        os.makedirs(dump_folder)

    epoched_test, _, ch_names, times = epoch_data(args.data_path, args.sub, args.n_ses, 'test', seed=20200220)
    epoched_train, img_conditions_train, _, _ = epoch_data(args.data_path, args.sub, args.n_ses, 'training', seed=20200220)

    print("length of test data:", len(epoched_test))
    print("length of train data:", len(epoched_train))
    print("Test data shape:", epoched_test[0].shape)
    print("Train data shape:", epoched_train[0].shape)
    
    split_and_dump(epoched_train, epoched_test, dump_folder, args.sub, ch_names, times, img_conditions_train, seed=20200220)