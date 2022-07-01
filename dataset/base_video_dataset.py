import torch.utils.data as Data
from utils.image_loader import opencv_loader

# base class for video datasets
class BaseVideoDataset(Data.Dataset):
    def __init__(self,name,root,image_loader=opencv_loader):
        """
        args:
            name-the name of dataset
            root-the root path to the dataset
            image_loader-the function to read the image
        """
        self.name=name
        self.root=root
        self.image_loader=image_loader

        self.sequence_list=[] 
        self.class_list=[]
        
    def __len__(self):
        return self.get_num_sequences()

    def __getitem__(self,index):
        """
        Not to be used.check get_frames() instead
        """
        return None

    def get_sequence_info(self,seq_id):
        """
        returns information about a particular sequences
        args:
            seq_id-index of the sequence
        returns:
            Dict
        """
        raise NotImplementedError

    def get_frames(self,seq_id,frame_ids,anno=None):
        """
        Get a set of frames from a particular sequence
        args:
            seq_id-index of sequence
            frame_ids-a list of frame numbers
            anno-the annotation for the sequence(see get_sequence_info).if None,they will be loaded.
        returns:
            list-List of frames corresponding to frame_ids
            list-List of dicts for each frame
            dict-A dict containing meta information about the sequence,e.g class of the target object.

        """
        raise NotImplementedError

    def is_video_sequence(self):
        # whether a video dataste or an image dataste
        return True

    def is_synthetic_video_dataset(self):
        # whether real videos or synthetic
        return False

    def get_name(self):
        # name of the dataset
        raise NotImplementedError

    def get_num_sequences(self):
        # number of sequences in a dataset
        return len(self.sequence_list)
    def has_class_info(self):
        return False

    def has_occlusion_info(self):
        return False

    def get_num_classes(self):
        return len(self.class_list)

    def get_class_list(self):
        return self.class_list

    def get_sequences_in_class(self, class_name):
        raise NotImplementedError

    def has_segmentation_info(self):
        return False