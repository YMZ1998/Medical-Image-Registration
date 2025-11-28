from moosez import moose

if __name__ == '__main__':
    input_file = r"D:\debug\LUNG1-226\IMAGES\CT.nii.gz"
    output_directory = r"D:\debug\LUNG1-226\IMAGES"

    moose(input_file, 'clin_ct_cardiac', output_directory, 'cuda')
