import tensorflow as tf
from config import config
import matplotlib.pyplot as plt

def compute_mfcc_tf(audio, sample_rate=16000):
    # audio: 1D tensor (float32), normalized between [-1, 1]

    # frame_length = 480      # 30 ms @ 16kHz, to set fft_length to 512 and not 1028
    # frame_step = 240        # 15 ms stride
    # fft_length = 512		# power of 2, efficient computation
    # num_mel_bins = 40
    # num_mfccs = 13			# standard for speech recognition
    # lower_edge_hertz = 20
    # upper_edge_hertz = 4000

    # compute STFT
    stft = tf.signal.stft(
        audio,
        frame_length=config.FRAME_LENGTH,
        frame_step=config.FRAME_STEP,
        fft_length=config.FFT_LENGTH
    )
    spectrogram = tf.abs(stft)   # no.of frames = (audio samples - frame_length) / frame_step + 1, frequency bins in each frame = fft_length / 2 + 1

    # create mel filterbank
    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=config.NUM_MEL_BINS,
        num_spectrogram_bins=config.FFT_LENGTH // 2 + 1,
        sample_rate=sample_rate,
        lower_edge_hertz=config.LOWER_EDGE_HERTZ,
        upper_edge_hertz=config.UPPER_EDGE_HERTZ,
    )
	
	# weighted sum of nearby frequency bins
    mel_spectrogram = tf.matmul(spectrogram, linear_to_mel_weight_matrix)

    # convert to log scale
    log_mel_spectrogram = tf.math.log(mel_spectrogram + 1e-6) #1e-6 added to prevent log(0)

    # compute mfccs
    mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)
    mfccs = mfccs[..., :config.NUM_MFCCS]

    return mfccs


def scheduler(epoch, lr):
    if epoch % 25 == 0 and epoch != 0:
        return lr * 0.1
    else:
        return lr


def find_lr_change(history):
    lr_h = history['lr']
    lr_change = {0: lr_h[0]}
    prev_lr = lr_h[0]

    for i in range(len(lr_h)):
        if prev_lr != lr_h[i]:
            lr_change[i] = lr_h[i]
            prev_lr = lr_h[i]
    return lr_change


def acc_loss_plot(history, path = None, size = [15,5]):
    lr_change = find_lr_change(history)

    fig, ax = plt.subplots(1,2, figsize=(size[0],size[1]))
    fig.tight_layout(pad=3)
    ax[0].plot(history['loss'], color='b', label="Training loss")
    ax[0].plot(history['val_loss'], color='r', label="Validation loss")
    for i in lr_change:
        ax[0].axvline(x=i, color='orange', linestyle='dotted', label=lr_change[i])
    ax[0].legend()
    ax[0].set_title('Loss plot')
    ax[0].set_xlabel('epochs')
    ax[0].set_ylabel('Loss')

    ax[1].plot(history['categorical_accuracy'], color='b', label="Training accuracy")
    ax[1].plot(history['val_categorical_accuracy'], color='r',label="Validation accuracy")
    for i in lr_change:
        ax[1].axvline(x=i, color='orange', linestyle='dotted', label=lr_change[i])
    ax[1].legend()
    ax[1].set_title('Accuracy plot')
    ax[1].set_xlabel('epochs')
    ax[1].set_ylabel('accuracy')

    if(path):
        plt.savefig(path, dpi = 300)
    plt.show()


def split_distribution(ds, ds_info, give_labels = False):
    class_labels = ds_info.features["label"].names
    classes = {}
    class_labels_list = []
    for class_label in class_labels:
        classes[class_label] = 0
    for (_, label) in ds:
        key = class_labels[label]
        classes[key] += 1
        if give_labels:
            class_labels_list.append(int(label))
    return classes, class_labels_list


def plot_distributions(train_dist, val_dist, test_dist, figsize=(25, 6)):
    fig, ax = plt.subplots(1, 3, figsize=figsize)
    fig.tight_layout(pad=3)
    ax[0].bar(list(train_dist.keys()), list(train_dist.values()))
    ax[0].set_title('Train Distribution')
    ax[0].set_xlabel('Classes')
    ax[0].set_ylabel('Count')

    ax[1].bar(list(val_dist.keys()), list(val_dist.values()))
    ax[1].set_title('Val Distribution')
    ax[1].set_xlabel('Classes')
    ax[1].set_ylabel('Count')

    ax[2].bar(list(test_dist.keys()), list(test_dist.values()))
    ax[2].set_title('Test Distribution')
    ax[2].set_xlabel('Classes')
    ax[2].set_ylabel('Count')