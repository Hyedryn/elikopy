import json

import os

rootdir = 'C:/Users/quent/Desktop/44cDTI/DTI/'
extensions1 = ('.json')
extensions2 = ('.gz', '.bval', '.bvec', '.json')

anonymize_json = False
rename = False

name_key = {}
for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        ext = os.path.splitext(file)[-1].lower()
        if ext in extensions1:
            print ('json: ', os.path.join(subdir, file))
            f = open(os.path.join(subdir, file),'r+')
            data = json.load(f)

            print(data.get('PatientID'))
            print()
            name_key.update({os.path.splitext(file)[0]:data.get('PatientID')})

            if anonymize_json:
                data["PatientName"] = data.get('PatientID')
                data["PatientBirthDate"] = data.get('PatientBirthDate')[:-3]
                f.seek(0)
                json.dump(data, f)
                f.truncate()
            f.close()
print()
print('Dict: ' + str(name_key))
print()
print()
for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        ext = os.path.splitext(file)[-1].lower()

        if ext in extensions2:
            print("Processing " + file)
            if ext in '.gz':
                ext = r'.nii.gz'
                ID = name_key.get(os.path.splitext(os.path.splitext(file)[0])[0])
            else:
                ID = name_key.get(os.path.splitext(file)[0])

            if ID:
                new_file = ID + "_DTI" + ext
                new_path = os.path.join(subdir, new_file)
                old_path = os.path.join(subdir, file)
                print("New path: " + new_path)
                print("Old path: "  + old_path)
                if rename:
                    os.rename(old_path,new_path)
            else:
                print("ID is none " + file)
                print(name_key)

            print()



            #



