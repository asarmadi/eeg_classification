import mne
import h5py
import argparse
import numpy as np
from utils.utils import stockwell
from mne.decoding import CSP
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

parser = argparse.ArgumentParser(description='EEG Classfication Network Testing')
parser.add_argument('--subject', default='1',type=str, help='Subject')
args = parser.parse_args()

def cal_acc(pred, target):
    total = len(pred)
    correct = (pred == target).sum()
    print(f"Accuracy: {correct}/{total}={100*correct/total}")
    return 100*correct/total


# Load EEG data from EDF files
file_path = './data/train_1d.h5'
with h5py.File(file_path, 'r',rdcc_nbytes=1024**2*4000,rdcc_nslots=1e7) as file:
  train_data     = np.array(file["data"][:])
  train_labels   = np.array(file["label"][:])  # 0 first thumb, 1 third thumb
  train_subjects = np.array(file["subject"][:])
  train_trial    = np.array(file["trial"][:])
  train_condition= np.array(file["condition"][:])

file_path = './data/test_1d.h5'
with h5py.File(file_path, 'r',rdcc_nbytes=1024**2*4000,rdcc_nslots=1e7) as file:
  test_data     = np.array(file["data"][:])
  test_labels   = np.array(file["label"][:])  # 0 first thumb, 1 third thumb
  test_subjects = np.array(file["subject"][:])
  test_trial    = np.array(file["trial"][:])
  test_condition= np.array(file["condition"][:])


# Define and apply CSP
n_components = 20
csp = CSP(n_components=n_components, reg=None, log=None, norm_trace=False, transform_into='csp_space')
csp.fit(train_data, train_labels)
X_train_csp = csp.transform(train_data)
X_test_csp  = csp.transform(test_data)
'''
fmax_samples = int(config.fmax*config.sig_time)
data_shape = (X_train_csp.shape[0], n_components, fmax_samples+1, config.window_len)
X_train_tra = np.zeros(data_shape)
for i in range(X_train_csp.shape[0]):
    X_train_tra[i] = stockwell(X_train_csp[i])

X_test_csp  = csp.transform(test_data)
data_shape = (X_test_csp.shape[0], n_components, fmax_samples+1, config.window_len)
X_test_tra  = np.zeros(data_shape)
for i in range(X_test_csp.shape[0]):
print(X_train_csp.shape, type(X_train_csp))
'''

f_data = h5py.File('./data/train_1d_csp.h5', "w")
f_data.create_dataset("data",      data = X_train_csp  )
f_data.create_dataset("label",     data = train_labels )
f_data.create_dataset("subject",   data = train_subjects)
f_data.create_dataset("trial",     data = train_trial )
f_data.create_dataset("condition", data = train_condition)
f_data.close()

f_data = h5py.File('./data/test_1d_csp.h5', "w")
f_data.create_dataset("data",      data = X_test_csp  )
f_data.create_dataset("label",     data = test_labels )
f_data.create_dataset("subject",   data = test_subjects)
f_data.create_dataset("trial",     data = test_trial )
f_data.create_dataset("condition", data = test_condition)
f_data.close()

exit(0)
clf    = MLPClassifier(random_state=1, max_iter=300).fit(X_train_csp, train_labels)
#clf = QDA().fit(X_train_csp, train_labels)
y_pred = clf.predict(X_train_csp)
train_acc = cal_acc(y_pred, train_labels)
# Apply CSP to test data and make predictions
y_test_pred = clf.predict(X_test_csp)
test_acc = cal_acc(y_test_pred, test_labels)

# Train LDA classifier
'''
lda = LDA()
lda.fit(X_train_csp, train_labels)
y_pred = lda.predict(X_train_csp)
train_acc = cal_acc(y_pred, train_labels)

# Apply CSP to test data and make predictions
X_test_csp  = csp.transform(test_data)
y_test_pred = lda.predict(X_test_csp)

test_acc = cal_acc(y_test_pred, test_labels)
'''

header_name = 'Name,Acc,Target'
data = [['Train', train_acc, args.subject], ['Test', test_acc, args.subject] ]

np.savetxt("./out/test_results_csp_"+args.subject+".csv", data, delimiter=",", header=header_name, comments='', fmt="%s")
