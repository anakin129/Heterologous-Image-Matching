import torch.utils.data as Data
class LTRLoader(Data.DataLoader):
    def __init__(self, name, dataset, training=True, batch_size=1, shuffle=False, num_workers=0):
        super(LTRLoader, self).__init__(dataset=dataset, batch_size=batch_size, shuffle=shuffle,num_workers=num_workers)
        self.name=name
        self.training=training

