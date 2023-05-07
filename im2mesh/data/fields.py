import os
import glob
import random
from PIL import Image
import numpy as np
import imageio
imageio.plugins.freeimage.download()

class ImagesField(object):
    ''' Image Field.

    It is the field used for loading images.

    Args:
        folder_name (str): image folder name
        mask_folder_name (str): mask folder name
        depth_folder_name (str): depth folder name
        visual_hull_depth_folder (str): visual hull depth folder name
        transform (transform): transformations applied to images
        extension (str): image extension
        mask_extension (str): mask extension
        depth_extension (str): depth extension
        with_camera (bool): whether camera data should be provided
        with_mask (bool): whether object masks should be provided
        with_depth (bool): whether depth maps should be provided
        random_view (bool): whether a random view should be used
        all_images (bool): whether all images should be returned (instead of
            one); only used for rendering
        n_views (int): number of views that should be used; if < 1, all views
            in the folder are used
        depth_from_visual_hull (bool): whether the visual hull depth map
            should be provided
        ignore_image_idx (list): list of IDs which should be ignored (only
            used for the multi-view reconstruction experiments)
    '''

    def __init__(self, folder_name, mask_folder_name='mask',
                 depth_folder_name='depth',
                 visual_hull_depth_folder='visual_hull_depth',
                 transform=None, extension='jpg', mask_extension='png',
                 depth_extension='exr', with_camera=False, with_mask=True,
                 with_depth=False, random_view=True,
                 all_images=False, n_views=0,
                 depth_from_visual_hull=False,
                 ignore_image_idx=[], **kwargs):
        self.folder_name = folder_name
        self.mask_folder_name = mask_folder_name
        self.depth_folder_name = depth_folder_name
        self.visual_hull_depth_folder = visual_hull_depth_folder

        self.transform = transform

        self.extension = extension
        self.mask_extension = mask_extension
        self.depth_extension = depth_extension

        self.random_view = random_view
        self.n_views = n_views

        self.with_camera = with_camera
        self.with_mask = with_mask
        self.with_depth = with_depth

        self.all_images = all_images

        self.depth_from_visual_hull = depth_from_visual_hull
        self.ignore_image_idx = ignore_image_idx

    def load(self, model_path, idx, category, input_idx_img=None):
        ''' Loads the field.

        Args:
            model_path (str): path to model
            idx (int): model id
            category (int): category id
            input_idx_img (int): image id which should be used (this
                overwrites any other id). This is used when the fields are
                cached.
        '''
        if self.all_images:
            n_files = self.get_number_files(model_path)
            data = {}
            for input_idx_img in range(n_files):
                datai = self.load_field(model_path, idx, category,
                                        input_idx_img)
                data['img%d' % input_idx_img] = datai
            data['n_images'] = n_files
            return data
        else:
            return self.load_field(model_path, idx, category, input_idx_img)

    def get_number_files(self, model_path, ignore_filtering=False):
        ''' Returns how many views are present for the model.

        Args:
            model_path (str): path to model
            ignore_filtering (bool): whether the image filtering should be
                ignored
        '''
        folder = os.path.join(model_path, self.folder_name)
        files = glob.glob(os.path.join(folder, '*.%s' % self.extension))
        files.sort()

        if not ignore_filtering and len(self.ignore_image_idx) > 0:
            files = [files[idx] for idx in range(
                len(files)) if idx not in self.ignore_image_idx]

        if not ignore_filtering and self.n_views > 0:
            files = files[:self.n_views]
        return len(files)

    def return_idx_filename(self, model_path, folder_name, extension, idx):
        ''' Loads the "idx" filename from the folder.

        Args:
            model_path (str): path to model
            folder_name (str): name of the folder
            extension (str): string of the extension
            idx (int): ID of data point
        '''
        folder = os.path.join(model_path, folder_name)
        files = glob.glob(os.path.join(folder, '*.%s' % extension))
        files.sort()

        if len(self.ignore_image_idx) > 0:
            files = [files[idx] for idx in range(
                len(files)) if idx not in self.ignore_image_idx]

        if self.n_views > 0:
            files = files[:self.n_views]
        return files[idx]

    def load_image(self, model_path, idx, data={}):
        ''' Loads an image.

        Args:
            model_path (str): path to model
            idx (int): ID of data point
            data (dict): data dictionary
        '''
        filename = self.return_idx_filename(model_path, self.folder_name,
                                            self.extension, idx)
        image = Image.open(filename).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        data[None] = image

    def load_camera(self, model_path, idx, data={}):
        ''' Loads an image.

        Args:
            model_path (str): path to model
            idx (int): ID of data point
            data (dict): data dictionary
        '''
        camera_file = os.path.join(model_path, 'cameras.npz')
        camera_dict = np.load(camera_file)

        if len(self.ignore_image_idx) > 0:
            n_files = self.get_number_files(model_path, ignore_filtering=True)
            idx_list = [i for i in range(
                n_files) if i not in self.ignore_image_idx]
            idx_list.sort()
            idx = idx_list[idx]

        camera_file = os.path.join(model_path, 'cameras.npz')
        camera_dict = np.load(camera_file)
        Rt = camera_dict['world_mat_%d' % idx].astype(np.float32)
        K = camera_dict['camera_mat_%d' % idx].astype(np.float32)
        S = camera_dict.get(
            'scale_mat_%d' % idx, np.eye(4)).astype(np.float32)
        data['world_mat'] = Rt
        data['camera_mat'] = K
        data['scale_mat'] = S

    def load_mask(self, model_path, idx, data={}):
        ''' Loads an object mask.

        Args:
            model_path (str): path to model
            idx (int): ID of data point
            data (dict): data dictionary
        '''
        filename = self.return_idx_filename(
            model_path, self.mask_folder_name, self.mask_extension, idx)
        mask = np.array(Image.open(filename)).astype(np.bool)
        mask = mask.reshape(mask.shape[0], mask.shape[1], -1)[:, :, 0]
        data['mask'] = mask.astype(np.float32)

    def load_depth(self, model_path, idx, data={}):
        ''' Loads a depth map.

        Args:
            model_path (str): path to model
            idx (int): ID of data point
            data (dict): data dictionary
        '''
        filename = self.return_idx_filename(
            model_path, self.depth_folder_name, self.depth_extension, idx)
        depth = np.array(imageio.imread(filename)).astype(np.float32)
        depth = depth.reshape(depth.shape[0], depth.shape[1], -1)[:, :, 0]
        data['depth'] = depth

    def load_visual_hull_depth(self, model_path, idx, data={}):
        ''' Loads a visual hull depth map.

        Args:
            model_path (str): path to model
            idx (int): ID of data point
            data (dict): data dictionary
        '''
        filename = self.return_idx_filename(
            model_path, self.visual_hull_depth_folder, self.depth_extension,
            idx)
        depth = np.array(imageio.imread(filename)).astype(np.float32)
        depth = depth.reshape(
            depth.shape[0], depth.shape[1], -1)[:, :, 0]
        data['depth'] = depth

    def load_field(self, model_path, idx, category, input_idx_img=None):
        ''' Loads the data point.

        Args:
            model_path (str): path to model
            idx (int): ID of data point
            category (int): index of category
            input_idx_img (int): image id which should be used (this
                overwrites any other id). This is used when the fields are
                cached.
        '''

        n_files = self.get_number_files(model_path)
        if input_idx_img is not None:
            idx_img = input_idx_img
        elif self.random_view:
            idx_img = random.randint(0, n_files - 1)
        else:
            idx_img = 0

        # Load the data
        data = {}
        self.load_image(model_path, idx_img, data)
        if self.with_camera:
            self.load_camera(model_path, idx_img, data)
        if self.with_mask:
            self.load_mask(model_path, idx_img, data)
        if self.with_depth:
            self.load_depth(model_path, idx_img, data)
        if self.depth_from_visual_hull:
            self.load_visual_hull_depth(model_path, idx_img, data)
        return data

    def check_complete(self, files):
        ''' Check if field is complete.

        Args:
            files: files
        '''
        complete = (self.folder_name in files)
        # TODO: check camera
        return complete

