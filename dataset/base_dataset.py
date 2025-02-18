from torch.utils.data import Dataset
import os


class BaseHSILoader:

    def load(self, *args, **kwargs):
        raise NotImplementedError


class BaseRGBLoader:

    def load(self, *args, **kwargs):
        raise NotImplementedError


class BaseSVDLoader:

    def load(self, *args, **kwargs):
        raise NotImplementedError


class BaseHyperDataset(Dataset):

    def __init__(self,
                 root_path='',
                 dataset='',
                 return_keys=None,
                 cache=True,
                 hsi_loader: BaseHSILoader = None,
                 rgb_loader: BaseRGBLoader = None,
                 svd_loader: BaseSVDLoader = None):

        if return_keys is None:
            return_keys = []
        self.dataset = dataset

        self.data_list = [
            {
                'root_path': root_path,
                'dataset': dataset,
                'name': name,
                'return_keys': return_keys,
                'from_which': 'cache' if cache else 'loader',
                'hsi_loader': hsi_loader,
                'rgb_loader': rgb_loader,
                'svd_loader': svd_loader,
                'hsi_cache': None,
                'rgb_cache': None,
                'svd_cache': None,
            } for name in self._get_names_from_path(root_path)
        ]

    def __len__(self):
        return len(self.data_list)

    def __add__(self, other: 'BaseHyperDataset'):
        added_dataset = BaseHyperDataset()
        added_dataset.dataset = f'{self.dataset}+{other.dataset}'
        added_dataset.data_list = self.data_list + other.data_list
        return added_dataset

    def __getitem__(self, idx):
        name = self.data_list[idx]['name']
        return_keys = self.data_list[idx]['return_keys']
        from_which = self.data_list[idx]['from_which']
        dataset = self.data_list[idx]['dataset']
        data_dict = {'name': name, 'dataset': dataset}

        for _k in return_keys:
            data_dict[_k] = self.get_data(idx, key=_k, from_which=from_which)

        return data_dict

    @staticmethod
    def _get_names_from_path(root_path):
        if not root_path:
            return []

        else:
            try:
                fns = os.listdir(os.path.join(root_path, 'hsi'))
            except FileNotFoundError:
                fns = os.listdir(os.path.join(root_path, 'rgb'))

            for i in range(len(fns)):
                extension = fns[i].rfind('.')
                if extension >= 0:
                    fns[i] = fns[i][:extension]

            return sorted(fns)

    def get_name(self, idx):
        return self.data_list[idx]['name']

    def get_data(self, idx, key, from_which='loader'):
        name = self.data_list[idx]['name']
        root_path = self.data_list[idx]['root_path']

        if from_which == 'loader':
            loader = self.data_list[idx][f'{key}_loader']
            if loader is None:
                return loader
            else:
                return loader.load(root_path, name)

        elif from_which == 'cache':
            cache = self.data_list[idx][f'{key}_cache']
            if cache is None:
                data = self.get_data(idx, key, from_which='loader')
                self.data_list[idx][f'{key}_cache'] = data
                return data
            else:
                return cache

        else:
            return None

    def get_hsi(self, idx, from_which='loader'):
        return self.get_data(idx, 'hsi', from_which)

    def get_rgb(self, idx, from_which='loader'):
        return self.get_data(idx, 'rgb', from_which)

    def get_svd(self, idx, from_which='loader'):
        return self.get_data(idx, 'svd', from_which)


