import os
import argparse

def main(path,train_p,val_p,test_p):
    images = sorted(os.listdir(os.path.join(path,"images")))
    n_images = len(images)

    n_training = int(n_images*train_p)
    n_validation = int(n_images*val_p)
    n_test = n_images - n_training - n_validation

    print("Total number of images: {}".format(n_images))
    print("Number of training images: {}".format(n_training))
    print("Number of validation images: {}".format(n_validation))
    print("Number of test images: {}".format(n_test))
    
    train_images = images[:n_training]
    val_images = images[n_training:n_training+n_validation]
    test_images = images[n_training+n_validation:]

    tr_file = open(os.path.join(path,"train.txt"),'w')
    for file in train_images:
        tr_file.write(os.path.join(path,"images",file))
        tr_file.write('\n')

    val_file = open(os.path.join(path,"val.txt"),'w')
    for file in val_images:
        val_file.write(os.path.join(path,"images",file))
        val_file.write('\n')

    test_file = open(os.path.join(path,"test.txt"),'w')
    for file in test_images:
        test_file.write(os.path.join(path,"images",file))
        test_file.write('\n')




if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-dir","--dataset_dir",help ="Dataset directory",type =str)
    parser.add_argument("-tr","--train_perc",help ="Training percetage",type =float)
    parser.add_argument("-val","--val_perc",help ="Validation percetage",type =float)
    parser.add_argument("-te","--test_perc",help ="Testing percetage",type =float)
    args = parser.parse_args()

    os.chdir(args.dataset_dir)
    main(os.getcwd(),args.train_perc,args.val_perc,args.test_perc)