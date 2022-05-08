import csv
from imutils import paths
import natsort
from numpy import genfromtxt
import pandas as pd
import numpy as np
from scipy.interpolate import griddata
from sklearn.preprocessing import scale


def data_marker(datasets, labels, image_size, frame_duration, overlap, locs_2d):
    Fs = 160.0  # sampling rate
    frame_length = Fs * frame_duration
    datasets = np.load(datasets)
    print('Generating training data...')

    for i in range(len(datasets)):
        print('Processing trial: ', i, '. (', i + 1, ' of ', len(datasets), ')')
        data = datasets[i].T
        df = pd.DataFrame(data)

        X_0 = aep_frame_maker(df, frame_duration)
        # steps = np.arange(0,len(df),frame_length)
        X_1 = X_0.reshape(len(X_0), 64 * 3)

        images = gen_images(np.array(locs_2d), X_1, image_size, normalize=False)
        images = np.swapaxes(images, 1, 3)
        print(len(images), ' frames generated with label ', labels[i], '.')
        print('\n')
        if i == 0:
            X = images
            y = np.ones(len(images)) * labels[0]
        else:
            X = np.concatenate((X, images), axis=0)
            y = np.concatenate((y, np.ones(len(images)) * labels[i]), axis=0)

    return X, np.array(y)


def aep_frame_maker(df, frame_duration):
    Fs = 160.0
    frame_length = Fs * frame_duration
    frames = []
    steps = steps_m(len(df), frame_duration, overlap)
    for i, _ in enumerate(steps):
        frame = []
        # if i == 0:
        # continue
        # else:
        for channel in df.columns:
            snippet = np.array(df.loc[steps[i][0]:steps[i][1], int(channel)])
            f, Y = fft(snippet)  # real part fft bul
            gama, alpha, beta = gama_alpha_beta_averages(f, Y)
            # plt.plot(f, Y)
            # plt.show()
            frame.append([Y.mean(), alpha, beta])
            # plt.plot(frame[0])
            # plt.show()

        # global sayac
        # if sayac==10:
        #     for k in frame:
        #       a=  sum(k)/3
        #       powerlist.append(a)
        # elif sayac==92:
        #     for k in frame:
        #         a = sum(k) / 3
        #         powerlist2.append(a)

        # sayac = sayac + 1

        frames.append(frame)
        # plt.plot(frames[0])
        # plt.show()
    return np.array(frames)


def gen_images(locs, features, n_gridpoints, normalize=True,
               augment=False, pca=False, std_mult=0.1, n_components=2, edgeless=False):
    """
    Generates EEG images given electrode locations in 2D space and multiple feature values for each electrode

    :param locs: An array with shape [n_electrodes, 2] containing X, Y
                        coordinates for each electrode.
    :param features: Feature matrix as [n_samples, n_features]
                                Features are as columns.
                                Features corresponding to each frequency band are concatenated.
                                (alpha1, alpha2, ..., beta1, beta2,...)
    :param n_gridpoints: Number of pixels in the output images
    :param normalize:   Flag for whether to normalize each band over all samples
    :param augment:     Flag for generating augmented images
    :param pca:         Flag for PCA based data augmentation
    :param std_mult     Multiplier for std of added noise
    :param n_components: Number of components in PCA to retain for augmentation
    :param edgeless:    If True generates edgeless images by adding artificial channels
                        at four corners of the image with value = 0 (default=False).
    :return:            Tensor of size [samples, colors, W, H] containing generated
                        images.
    """
    feat_array_temp = []
    nElectrodes = locs.shape[0]  # Number of electrodes
    # Test whether the feature vector length is divisible by number of electrodes
    assert features.shape[1] % nElectrodes == 0
    n_colors = features.shape[1] // nElectrodes
    for c in range(int(n_colors)):
        feat_array_temp.append(features[:, c * nElectrodes: nElectrodes * (c + 1)])

    nSamples = features.shape[0]
    # Interpolate the values
    grid_x, grid_y = np.mgrid[
                     min(locs[:, 0]):max(locs[:, 0]):n_gridpoints * 1j,
                     min(locs[:, 1]):max(locs[:, 1]):n_gridpoints * 1j
                     ]
    temp_interp = []
    for c in range(n_colors):
        temp_interp.append(np.zeros([nSamples, n_gridpoints, n_gridpoints]))
    # Generate edgeless images
    if edgeless:
        min_x, min_y = np.min(locs, axis=0)
        max_x, max_y = np.max(locs, axis=0)
        locs = np.append(locs, np.array([[min_x, min_y], [min_x, max_y], [max_x, min_y], [max_x, max_y]]), axis=0)
        for c in range(n_colors):
            feat_array_temp[c] = np.append(feat_array_temp[c], np.zeros((nSamples, 4)), axis=1)
    # Interpolating
    for i in range(nSamples):
        for c in range(n_colors):
            temp_interp[c][i, :, :] = griddata(locs, feat_array_temp[c][i, :], (grid_x, grid_y),
                                               method='cubic', fill_value=np.nan)
        print('Interpolating {0}/{1}\r'.format(i + 1, nSamples), end='\r')
    # Normalizing
    for c in range(n_colors):
        if normalize:
            temp_interp[c][~np.isnan(temp_interp[c])] = \
                scale(temp_interp[c][~np.isnan(temp_interp[c])])
        temp_interp[c] = np.nan_to_num(temp_interp[c])
    return np.swapaxes(np.asarray(temp_interp), 0, 1)  # swap axes to have [samples, colors, W, H]


def steps_m(samples, frame_duration, overlap):
    Fs = 160
    i = 0
    intervals = []
    samples_per_frame = Fs * frame_duration
    while i + samples_per_frame <= samples:
        intervals.append((i, i + samples_per_frame))
        i = i + samples_per_frame - int(samples_per_frame * overlap)
    return intervals


def fft(snippet):
    Fs = 160.0  # sampling rate
    # Ts = len(snippet)/Fs/Fs; # sampling interval
    snippet_time = len(snippet) / Fs
    Ts = 1.0 / Fs  # sampling interval
    t = np.arange(0, snippet_time, Ts)  # time vector

    # ff = 5;   # frequency of the signal
    # y = np.sin(2*np.pi*ff*t)
    y = snippet
    #     print('Ts: ',Ts)
    #     print(t)
    #     print(y.shape)
    n = len(y)  # length of the signal
    k = np.arange(n)
    T = n / Fs
    frq = k / T  # two sides frequency range
    frq = frq[range(n // 2)]  # one side frequency range

    #   Y = np.fft.fft(y)/n # fft computing and normalization
    # ydeneme=np.fft.fft(y)

    Y = np.fft.fft(y)
    Y = abs(Y)
    # Y=np.square(Y)
    Y = Y / n

    Y = Y[range(n // 2)]
    # ydeneme =ydeneme[range(n // 2)]
    # plt.plot(frq, Y)
    # plt.show()
    # plt.plot(frq, ydeneme)
    # plt.show()
    # plt.plot(frq, abs(Y))
    # plt.show()

    # Added in: (To remove bias.)
    # Y[0] = 0
    # return frq,abs(Y)
    return frq, Y


def gama_alpha_beta_averages(f, Y):
    gama_range = (30, 45)
    alpha_range = (8, 13)
    beta_range = (14, 30)
    # gama1 = Y[(f > gama_range[0]) & (f <= gama_range[1])].sum()
    gama = Y[(f > gama_range[0]) & (f <= gama_range[1])].mean()
    alpha = Y[(f > alpha_range[0]) & (f <= alpha_range[1])].mean()
    beta = Y[(f > beta_range[0]) & (f <= beta_range[1])].mean()

    return gama, alpha, beta


if __name__ == "__main__":
    # sira = '52'
    resim_boyut = 32
    dataset = '/home/teslav/DataSets/EEGMMIDB/data.npy'
    label = '/home/teslav/DataSets/EEGMMIDB/label.npy'
    frame_duration = 1
    overlap = 0  # degisecek
    # batch_size = 64
    # num_classes = 2
    # epochs = 400
    # model_save = 'modeller/m' + sira
    # test_sonuc = "sonuclar/sonuc" + sira
    # test_sonuc2 = "sonuclar/confision" + sira
    # PR = "sonuclar/PR-Grafik" + sira
    # roc = "sonuclar/roc" + sira

    # imagePaths = sorted(list(paths.list_files(dataset)))
    # imagePaths = (natsort.natsorted(imagePaths))
    # print(imagePaths)
    # datasets = imagePaths

    # with open(etiket) as f:
    #     output = [float(s) for line in f.readlines() for s in line[:-1].split(',')]
    #     output = [round(x) for x in output]
        # print(output)

    # labels = output
    labels = np.load(label)
    image_size = resim_boyut

    # location read
    results = []
    with open("/home/teslav/DataSets/EEGMMIDB/BioSemi64.loc") as file:
        # reader = csv.reader(file, quoting=csv.QUOTE_NONNUMERIC)  # change contents to floats
        # for row in reader:  # each row is a list
        #     results.append(np.array(row))
        #     # print(row)
        for line in file.readlines():
            elem = line.split('\t')
            results.append([float(elem[1]), float(elem[2])])

    locs_2d = np.array(results)

    X, y = data_marker(dataset, labels, image_size, frame_duration, overlap, locs_2d)
    np.save('/home/teslav/DataSets/EEGMMIDB/EEG2Img/data.npy', X)
    np.save('/home/teslav/DataSets/EEGMMIDB/EEG2Img/label.npy', y)
    print(X.shape)
